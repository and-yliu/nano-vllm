from dataclasses import dataclass 
import torch 


# cu_seqlens: cumulative sequence lengths, describe where each variable-length sequence begins and ends after sequences are packed into one tensor.

@dataclass
class Context:
    is_prefill: bool = False                        # whether entire prompts are currently being processed.
    cu_seqlens_q: torch.Tensor | None = None        # boundaries of packed query sequences.
    cu_seqlens_k: torch.Tensor | None = None        # boundaries of packed key/value sequences.
    max_seqlen_q: int = 0                           # longest query sequence in the batch.
    max_seqlen_k: int = 0                           # longest key sequence in the batch.
    slot_mapping: torch.Tensor | None = None        # where new key/value entries should be written in the KV cache.
    context_lens: torch.Tensor | None = None        # how many cached tokens each sequence currently has.
    block_tables: torch.Tensor | None = None        # which physical KV-cache blocks belong to each sequence.

_context = Context()

def get_context() -> Context:
    return _context

def reset_context():
    global _context
    _context = Context()

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _context
    _context = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)