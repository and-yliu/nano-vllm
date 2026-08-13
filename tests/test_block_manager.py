"""BlockManager: paged KV allocation and prefix-cache reuse.

    python -m unittest tests.test_block_manager -v
"""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minivllm.engine.block_manager import BlockManager
from minivllm.engine.sequence import Sequence

BLOCK_SIZE = 4


def make_seq(token_ids: list[int]) -> Sequence:
    return Sequence(token_ids, block_size=BLOCK_SIZE)


def admit(bm: BlockManager, seq: Sequence) -> int:
    """Probe-then-allocate, as the scheduler does. Returns the blocks hit."""
    num_cached_blocks = bm.can_allocate(seq)
    assert num_cached_blocks != -1, "sequence did not fit"
    bm.allocate(seq, num_cached_blocks)
    return num_cached_blocks


class BlockManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bm = BlockManager(num_blocks=16, block_size=BLOCK_SIZE)

    # -- allocation basics ------------------------------------------------

    def test_allocate_builds_block_table_and_consumes_pool(self) -> None:
        seq = make_seq(list(range(10)))          # 3 blocks: 4 + 4 + 2
        self.assertEqual(seq.num_blocks, 3)

        admit(self.bm, seq)

        self.assertEqual(len(seq.block_table), 3)
        self.assertEqual(len(self.bm.free_block_ids), 13)
        self.assertEqual(self.bm.used_block_ids, set(seq.block_table))
        # nothing was cached, so nothing may be claimed as cached
        self.assertEqual(seq.num_cached_tokens, 0)

    def test_block_table_holds_ids_not_objects(self) -> None:
        """Both allocate loops must append ints -- the runner indexes with them."""
        seq = make_seq(list(range(10)))
        admit(self.bm, seq)
        for block_id in seq.block_table:
            self.assertIsInstance(block_id, int)

    def test_deallocate_returns_blocks_to_pool(self) -> None:
        seq = make_seq(list(range(10)))
        admit(self.bm, seq)
        self.bm.deallocate(seq)

        self.assertEqual(len(self.bm.free_block_ids), 16)
        self.assertEqual(self.bm.used_block_ids, set())
        self.assertEqual(seq.block_table, [])
        self.assertEqual(seq.num_cached_tokens, 0)

    def test_partial_block_is_never_cached(self) -> None:
        """A trailing partial block's contents still change, so it must not hash."""
        seq = make_seq(list(range(6)))           # one full block + 2 tokens
        admit(self.bm, seq)

        full, partial = seq.block_table
        self.assertNotEqual(self.bm.blocks[full].hash, -1)
        self.assertEqual(self.bm.blocks[partial].hash, -1)
        # only the full block is published
        self.assertEqual(list(self.bm.hash_to_block_id.values()), [full])

    def test_miss_blocks_are_published_for_later_reuse(self) -> None:
        """allocate must hash the blocks it allocates fresh, not just the hits.

        Skipping this both kills cross-request reuse of prompt blocks and leaves
        a boundary-length prompt with an unhashed last block, which trips
        append()'s `hash != -1` assert on the first decode token.
        """
        seq = make_seq(list(range(8)))           # exactly 2 full blocks
        admit(self.bm, seq)

        for block_id in seq.block_table:
            block = self.bm.blocks[block_id]
            self.assertNotEqual(block.hash, -1, "full block left unpublished")
            self.assertEqual(self.bm.hash_to_block_id[block.hash], block_id)

        seq.append_token(9)                      # crosses the boundary
        self.assertTrue(self.bm.can_append(seq))
        self.bm.append(seq)                      # must not assert
        self.assertEqual(len(seq.block_table), 3)

    # -- can_allocate contract --------------------------------------------

    def test_can_allocate_signals_failure_with_minus_one(self) -> None:
        """0 hits is success; -1 is refusal. Both are truthy -- callers must
        compare against -1, so the two cases have to stay distinguishable."""
        bm = BlockManager(num_blocks=3, block_size=BLOCK_SIZE)

        exact = make_seq(list(range(12)))        # exactly 3 blocks
        self.assertEqual(bm.can_allocate(exact), 0, "must not be off by one")

        too_big = make_seq(list(range(13)))      # 4 blocks
        self.assertEqual(bm.can_allocate(too_big), -1)

    def test_can_allocate_discounts_shared_blocks(self) -> None:
        """A hit on a live block costs nothing from the free pool."""
        bm = BlockManager(num_blocks=3, block_size=BLOCK_SIZE)
        tokens = list(range(8))                  # 2 blocks

        first = make_seq(tokens)
        admit(bm, first)                         # holds 2 of 3 blocks
        self.assertEqual(len(bm.free_block_ids), 1)

        second = make_seq(tokens)
        # block 0 is shared, so only block 1 needs the last free block
        self.assertEqual(bm.can_allocate(second), 1)
        admit(bm, second)
        self.assertEqual(second.block_table[0], first.block_table[0])
        self.assertEqual(len(bm.free_block_ids), 0)

    def test_can_allocate_charges_for_revived_blocks(self) -> None:
        """A hit on a *freed* block still pulls it out of the free list."""
        bm = BlockManager(num_blocks=2, block_size=BLOCK_SIZE)
        tokens = list(range(8))

        first = make_seq(tokens)
        admit(bm, first)
        bm.deallocate(first)                     # both blocks free, both cached

        second = make_seq(tokens)
        self.assertEqual(bm.can_allocate(second), 1, "block 0 hits, but is not free-of-charge")
        admit(bm, second)
        self.assertEqual(len(bm.free_block_ids), 0)

    def test_last_block_is_never_probed(self) -> None:
        """Deliberate: the probe stops at num_blocks - 1 so the forward pass
        always has at least one block of work to do."""
        tokens = list(range(12))                 # 3 full blocks
        first = make_seq(tokens)
        admit(self.bm, first)
        self.bm.deallocate(first)

        second = make_seq(tokens)
        self.assertEqual(self.bm.can_allocate(second), 2, "last block must not count as a hit")
        admit(self.bm, second)
        self.assertEqual(second.num_cached_tokens, 8)
        self.assertLess(second.num_cached_tokens, second.num_tokens, "must leave work to compute")

    # -- prefix cache -----------------------------------------------------

    def test_reuse_across_finished_sequences(self) -> None:
        """The point of the cache: request 2 skips prefill for request 1's prefix."""
        tokens = list(range(12))                 # 3 full blocks

        first = make_seq(tokens)
        admit(self.bm, first)
        first_table = list(first.block_table)
        self.bm.deallocate(first)

        second = make_seq(tokens)
        admit(self.bm, second)

        self.assertEqual(second.block_table[:2], first_table[:2], "should revive the same blocks")
        self.assertEqual(second.num_cached_tokens, 8, "all but the last block was cached")

    def test_revive_preserves_block_identity(self) -> None:
        """A revived block must stay hittable -- allocation resets it internally."""
        tokens = list(range(12))

        first = make_seq(tokens)
        admit(self.bm, first)
        table = list(first.block_table)
        self.bm.deallocate(first)

        second = make_seq(tokens)                # revives blocks 0 and 1
        admit(self.bm, second)

        for block_id in table[:2]:
            block = self.bm.blocks[block_id]
            self.assertNotEqual(block.hash, -1, "revived block lost its hash")
            self.assertNotEqual(block.token_ids, [], "revived block lost its tokens")
            self.assertEqual(self.bm.hash_to_block_id[block.hash], block_id)

        # and a third sequence must still be able to hit them
        third = make_seq(tokens)
        admit(self.bm, third)
        self.assertEqual(third.block_table[:2], table[:2])
        self.assertEqual(third.num_cached_tokens, 8)

    def test_concurrent_sequences_share_blocks_by_refcount(self) -> None:
        tokens = list(range(12))
        first, second = make_seq(tokens), make_seq(tokens)

        admit(self.bm, first)
        admit(self.bm, second)                   # first is still live

        shared = first.block_table[:2]
        self.assertEqual(second.block_table[:2], shared)
        for block_id in shared:
            self.assertEqual(self.bm.blocks[block_id].ref_count, 2)

        # freeing one must not release shared blocks
        self.bm.deallocate(first)
        self.assertEqual(self.bm.used_block_ids, set(second.block_table))
        for block_id in second.block_table:
            self.assertEqual(self.bm.blocks[block_id].ref_count, 1)

        self.bm.deallocate(second)
        self.assertEqual(self.bm.used_block_ids, set())

    def test_shared_prefix_with_divergent_suffix(self) -> None:
        """Hits stop at the first differing block, and only then."""
        first = make_seq(list(range(12)))
        admit(self.bm, first)

        second = make_seq(list(range(4)) + [99, 98, 97, 96] + [5, 5, 5, 5])
        admit(self.bm, second)

        self.assertEqual(second.block_table[0], first.block_table[0], "prefix shared")
        self.assertNotEqual(second.block_table[1], first.block_table[1], "suffix differs")
        self.assertEqual(second.num_cached_tokens, 4, "only the first block was cached")

    def test_same_tokens_different_prefix_do_not_collide(self) -> None:
        """Chaining means block content alone must not produce a hit."""
        first = make_seq([1, 1, 1, 1] + [7, 7, 7, 7])
        admit(self.bm, first)

        second = make_seq([2, 2, 2, 2] + [7, 7, 7, 7])
        admit(self.bm, second)

        self.assertNotEqual(
            second.block_table[1], first.block_table[1],
            "identical tokens under a different prefix must not share a block",
        )
        self.assertEqual(second.num_cached_tokens, 0)

    def test_evicted_block_does_not_serve_stale_hit(self) -> None:
        """Once a block is reused for other data, its old hash must not match."""
        bm = BlockManager(num_blocks=2, block_size=BLOCK_SIZE)

        first = make_seq([1, 2, 3, 4, 5, 6, 7, 8])
        admit(bm, first)
        stale_hash = bm.blocks[first.block_table[0]].hash
        bm.deallocate(first)

        # only two blocks exist, so this must recycle the first sequence's data
        second = make_seq([9, 10, 11, 12, 13, 14, 15, 16])
        admit(bm, second)
        self.assertEqual(second.num_cached_tokens, 0)
        bm.deallocate(second)

        third = make_seq([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(
            bm.can_allocate(third), 0,
            f"stale hash {stale_hash} served a block whose contents were overwritten",
        )

    def test_recycled_block_drops_its_index_entry(self) -> None:
        """Otherwise hash_to_block_id grows without bound over a long run."""
        bm = BlockManager(num_blocks=2, block_size=BLOCK_SIZE)

        first = make_seq([1, 2, 3, 4, 5, 6, 7, 8])
        admit(bm, first)
        stale_hash = bm.blocks[first.block_table[0]].hash
        bm.deallocate(first)

        second = make_seq([9, 10, 11, 12, 13, 14, 15, 16])
        admit(bm, second)
        self.assertNotIn(stale_hash, bm.hash_to_block_id)

    # -- decode-time growth -----------------------------------------------

    def test_append_finalizes_hash_when_block_fills(self) -> None:
        seq = make_seq([1, 2, 3])                # one partial block
        admit(self.bm, seq)
        block_id = seq.block_table[0]
        self.assertEqual(self.bm.blocks[block_id].hash, -1)

        seq.append_token(4)                      # num_tokens % 4 == 0 -> now full
        self.assertTrue(self.bm.can_append(seq))
        self.bm.append(seq)

        block = self.bm.blocks[block_id]
        self.assertNotEqual(block.hash, -1, "full block should be published")
        self.assertEqual(self.bm.hash_to_block_id[block.hash], block_id)
        self.assertEqual(len(seq.block_table), 1, "no new block needed yet")

    def test_append_allocates_when_crossing_boundary(self) -> None:
        seq = make_seq([1, 2, 3, 4])
        admit(self.bm, seq)

        seq.append_token(5)                      # num_tokens % 4 == 1 -> needs a block
        self.assertTrue(self.bm.can_append(seq))
        free_before = len(self.bm.free_block_ids)
        self.bm.append(seq)

        self.assertEqual(len(seq.block_table), 2)
        self.assertEqual(len(self.bm.free_block_ids), free_before - 1)
        for block_id in seq.block_table:
            self.assertIsInstance(block_id, int)

    def test_can_append_false_when_pool_exhausted(self) -> None:
        bm = BlockManager(num_blocks=1, block_size=BLOCK_SIZE)
        seq = make_seq([1, 2, 3, 4])
        admit(bm, seq)

        seq.append_token(5)                      # would need a second block
        self.assertFalse(bm.can_append(seq), "no free blocks left, must signal preempt")

    def test_growth_then_reuse_end_to_end(self) -> None:
        """Decode-grown blocks must be reusable by a later request."""
        seq = make_seq([1, 2, 3, 4])
        admit(self.bm, seq)
        for token in (5, 6, 7, 8):
            seq.append_token(token)
            self.assertTrue(self.bm.can_append(seq))
            self.bm.append(seq)
        self.bm.deallocate(seq)

        later = make_seq([1, 2, 3, 4, 5, 6, 7, 8])
        admit(self.bm, later)
        self.assertEqual(
            later.num_cached_tokens, 4,
            "blocks filled during decode should be reusable (last block never probed)",
        )


if __name__ == "__main__":
    unittest.main()
