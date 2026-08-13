from collections import deque
from minivllm.engine.sequence import Sequence, SequenceStatus
from minivllm.engine.block_manager import BlockManager

class Scheduler:
    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int = -1):
        self.waiting: deque[Sequence] = deque()                                          # arrival order
        self.running: deque[Sequence] = deque()                                          # currently holding blocks
        self.block_manager: BlockManager = BlockManager(max_cached_blocks, block_size)
        self.block_size: int = block_size                                                # tokens per KV block
        self.max_num_seqs: int = max_num_seqs                                            # batch width cap
        self.max_num_batched_tokens: int = max_num_batched_tokens                        # prefill work cap per step
        self.eos_token_id: int = eos

    # add a seq to waiting queue
    def add(self, seq: Sequence):
        # Reject up front what the scheduler could never satisfy, otherwise the
        # sequence sits in `waiting` and only surfaces later as a stalled engine.
        capacity = len(self.block_manager.blocks)
        if seq.num_blocks > capacity:
            raise ValueError(
                f"sequence {seq.seq_id} needs {seq.num_blocks} blocks "
                f"({seq.num_tokens} tokens at block_size={self.block_size}) but the KV "
                f"cache only holds {capacity}; raise gpu_memory_utilization or "
                "shorten the prompt"
            )
        if seq.num_tokens > self.max_num_batched_tokens:
            raise ValueError(
                f"sequence {seq.seq_id} needs {seq.num_tokens} tokens of prefill but "
                f"max_num_batched_tokens is {self.max_num_batched_tokens}; prefill is "
                "not chunked, so raise the budget or shorten the prompt"
            )
        self.waiting.append(seq)

    # if waiting queue or running is not empty
    def has_work(self) -> bool:
        return len(self.waiting) > 0 or len(self.running) > 0

    # schedule the next batch of sequence to run
    def schedule(self) -> tuple[list[Sequence], bool]: # (seqs, is_prefill)
        # batch of sequence
        scheduled_seqs = []
        # current number of tokens
        num_batched_tokens = 0

        preempted = False

        # first do all prefill tasks
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            # check the remaining token budgets
            remaining = self.max_num_batched_tokens - num_batched_tokens
            # getting the number of token to precess depending if the sequence has been allocated
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # Prefill is all-or-nothing: the runner computes the whole uncached
            # suffix in one pass, so a seq that does not fit in what is left of
            # the budget waits for a step of its own. Scheduling it partially
            # would leave it at the head of `waiting` with nothing to advance it.
            # add() has already rejected seqs that can never fit, so breaking
            # here always makes progress on some later step.
            if remaining < num_tokens:
                break

            # allocate the blocks; this also advances seq.num_cached_tokens over
            # whatever the prefix cache hit. can_allocate only matches *full*
            # blocks and never the last one, so at least one token is always
            # left for this pass to compute.
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = num_tokens
            num_batched_tokens += num_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # if not prefill task, get decode tasks
        while self.running:
            # get the first running deque
            seq = self.running.popleft()
            # check if the additional token can be appended and store in a block
            if not self.block_manager.can_append(seq):
                # if not, pop and clear the last seq that started to run and 
                preempted = True
                if self.running:
                    self.running.appendleft(seq)
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                # if it can be run, but the batch token or max number of sequence exceed limit
                if num_batched_tokens >= self.max_num_batched_tokens or len(scheduled_seqs) >= self.max_num_seqs:
                    # place it first in running queue
                    self.running.appendleft(seq)
                    break

                # append addition block if needed
                self.block_manager.append(seq)
                # add seq to the list that is pending process
                scheduled_seqs.append(seq)
                seq.num_scheduled_tokens = 1

        if scheduled_seqs:
            # extendleft push the list in one by one on the left, which require reverse to have the previous order
            self.running.extendleft(reversed(scheduled_seqs))
        elif not preempted and (self.waiting or self.running):
            raise RuntimeError(
                "Scheduler made no progress: "
                f"{len(self.waiting)} waiting and {len(self.running)} running sequences, "
                f"{len(self.block_manager.free_block_ids)} of "
                f"{len(self.block_manager.blocks)} blocks free. "
                "This means either a sequence that cannot fit in the KV cache, or "
                "blocks leaked because their ref_count never returned to 0."
            )

        return scheduled_seqs, False

    # clear and empty a sequence, put it first in the waiting queue
    def preempt(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)

    # after process to see if the sequnece should be set to the FINISHED state
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)

            # if last_token is end of sentence
            stop_due_to_eos = not seq.sampling_params.ignore_eos and token_id == self.eos_token_id
            # if output tokens exceed max_tokens
            stop_due_to_max_tokens = seq.num_output_tokens >= seq.sampling_params.max_tokens
            # if total length exceed max_model_length
            stop_due_to_max_length = seq.sampling_params.max_model_length is not None and seq.num_tokens >= seq.sampling_params.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
    