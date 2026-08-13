from enum import Enum, auto
import math
from itertools import count 
from minivllm.sampling_params import SamplingParams
from copy import copy

class SequenceStatus(Enum):
    WAITING = auto()    # admitted to the queue, no KV blocks yet
    RUNNING = auto()    # has blocks, being prefilled or decoded
    FINISHED = auto()   # hit EOS or max_tokens; blocks released

class Sequence:
    """One request: its tokens, its KV blocks, and how much of it is cached.

    The engine's core invariant is `num_cached_tokens`: the number of leading
    tokens whose K/V are already written into the paged cache. Everything the
    scheduler and model runner do is expressed relative to it --
        prefill  = compute tokens [num_cached_tokens, num_tokens)
        decode   = compute exactly one token, the last
    so a sequence that is partially cached (preempted and resumed, or later,
    prefix-cached) needs no special casing.
    """
    _counter = count()

    def __init__(self, token_ids: list[int], block_size: int, sampling_params = SamplingParams()):
        self.seq_id = next(Sequence._counter)
        self.token_ids = list(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.sampling_params = sampling_params or SamplingParams()

        self.status = SequenceStatus.WAITING
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.block_table: list[int] = []
        # number of token per block
        self.block_size = block_size

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED
    
    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_output_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    @property
    def num_cached_blocks(self):
        return int(math.ceil(self.num_cached_tokens / self.block_size))

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i: int):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i+1)*self.block_size]