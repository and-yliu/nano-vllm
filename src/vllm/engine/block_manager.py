import xxhash
import numpy as np
from collections import deque

from vllm.engine.sequence import Sequence

class Block:
    def __init__(self, block_id):
        self.block_id = block_id    
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []

    def update(self, h: int, token_ids: list[int]):
        self.hash = h 
        self.token_ids = token_ids

    def reset(self):
        self.hash = -1
        self.ref_count = 1
        self.token_ids = []

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]

        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        h = xxhash.xxh64()
        if prefix_hash_value != -1:
            h.update(prefix_hash_value.to_bytes(8, 'little'))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    # allocate a free block and set it to used
    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        # reset block to initial state and having 1 ref_count
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    # deallocate a used block and set it to free
    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # check if there are enough block for this sequence, if not -1, else number of cache hit
    def can_allocate(self, seq: Sequence):
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks

        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # if no hash is store, the break and return
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            # count the number of cached block
            num_cached_blocks += 1
            # if it is in use, the we can allocate one less block
            if block_id in self.used_block_ids:
                num_new_blocks -= 1

        # if the number of block need is more than the number of free block, return -1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        
        return num_cached_blocks

    # called when a sequence is moving from WAITING to RUNNING
    # build the block_table of the sequence
    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table, "allocate() called on a sequence that already holds blocks"

        h = -1
        # for all cached hits, append the block id to the block table
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)

        # for not cached block, allocate new blocks at set their hash and tokens
        for i in range(num_cached_blocks, seq.num_blocks):
            token_ids = seq.block(i)
            block_id = self._allocate_block()
            if len(token_ids) == self.block_size:
                h = self.compute_hash(token_ids, h)      # h carried from loop 1
                self.blocks[block_id].update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            else:
                h = -1
            seq.block_table.append(block_id)

    # called when a sequence is on FINISHED
    def deallocate(self, seq: Sequence) -> None:
        # update block information
        for block_id in seq.block_table:
            block = self.blocks[block_id]

            # drop ref from every block of the sequence, if ref is 0, move it to free pool.
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
    
        # update sequence information
        seq.block_table = []
        seq.num_cached_tokens = 0

    # check if the next token can be successfully appended to the current block or an next block
    def can_append(self, seq: Sequence) -> bool:
        if seq.num_tokens % self.block_size == 1:
            return len(self.free_block_ids) > 0
        return True    

    # for each decoder step, to see if a block is full or if a new block is needed
    def append(self, seq: Sequence) -> None:
        block_tables = seq.block_table

        last_block_for_seq_id = block_tables[-1]

        # last block is just full after the additional token
        if seq.num_tokens % self.block_size == 0:
            h = self.compute_hash(token_ids=seq.block(seq.num_blocks - 1), prefix_hash_value = -1 if len(block_tables) == 1 else self.blocks[block_tables[-2]].hash)
            block = self.blocks[last_block_for_seq_id]
            block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))
            self.hash_to_block_id[h] = block.block_id
        # need a new block for the new token
        elif seq.num_tokens % self.block_size == 1:
            assert self.blocks[last_block_for_seq_id].hash != -1
            block_id = self._allocate_block()
            block_tables.append(block_id)
        # last block already allocated and block is not full
        else:
            assert last_block_for_seq_id in self.used_block_ids, "Last block should be allocated"
            assert self.blocks[last_block_for_seq_id].hash == -1, "Last block should be partial block with hash -1"
