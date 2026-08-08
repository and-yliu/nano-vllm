import torch
import torch.nn as nn

# RoPE (Rotary Positional Embedding)
# RoPE encodes token position by **rotating the Query (Q) and Key (K) vectors** based on their position before attention.

# Split each Q/K vector into a pairs of two and rotate each pair by a position-dependent angle.
# Different dimension pairs use different rotation frequencies, allowing the model to capture both short- and long-range positional relationships.
# When attention computes (Q^TK), the rotations combine so that the score depends on the relative position (q-p) between tokens.
#(R_p @ ​Q)^T @ (R_q @ ​K) = Q^T @ R_p^T @ R_q @ ​K = Q^T @ R_-p @ R_q @ ​K = Q^T @ R_q-p @ ​K 


def add_rotary_embedding(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 3:
        seq_len, head_size, head_dim = tensor.shape

        #(seq_len, head_size, head_dim/2)
        t1, t2 = torch.chunk(tensor, 2, dim=-1)

        # cos -> (seq_len, 1, head_dim/2) 
        # sin -> (seq_len, 1, head_dim/2)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        x1 = t1 * cos - t2 * sin
        x2 = t1 * sin + t2 * cos

        #(seq_len, head_size, head_dim)
        return torch.cat([x1, x2], dim=-1)
    else:
        batch, seq_len, head_size, head_dim = tensor.shape

        #(batch, seq_len, head_size, head_dim/2)
        t1, t2 = torch.chunk(tensor, 2, dim=-1)

        # cos -> (1, seq_len, 1, head_dim/2) 
        # sin -> (1, seq_len, 1, head_dim/2)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)


        x1 = t1 * cos - t2 * sin
        x2 = t1 * sin + t2 * cos

        #(batch, seq_len, head_size, head_dim)
        return torch.cat([x1, x2], dim=-1)
        

class RotaryEmbedding(nn.Module):
    def __init__(self, base: int, embed_dim: int, max_position: int = 2024):
        super().__init__()
        self.base = base
        # the number of dimension to apply rotary embedding
        self.embed_dim = embed_dim
        # the maximum context length, the model supports
        self.max_position = max_position
        # different frequencey to make for each pair of embedding
        self.inv_freq = 1.0/ (base ** (torch.arange(0, embed_dim, 2)/embed_dim))
        tokens = torch.arange(0, max_position, dtype=float) 

        #  the angle inputs used to build the rotary embeddings based on position and freq
        freqs = tokens[:, None] * self.inv_freq[None, :]
        # radian vlaues
        cos_cached = torch.cos(freqs)
        sin_cached = torch.sin(freqs)

        # (max_position, embed_dim)
        rotary_cached = torch.cat([cos_cached, sin_cached], dim=-1)
        # store them
        self.register_buffer("cos_sin_cache", rotary_cached)

    @torch.compile
    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # retrieve cos and sin radiants 
        cos_sin = self.cos_sin_cache[positions]     #(seq_len, embed_dim)
        cos, sin = torch.chunk(cos_sin, 2, dim=-1)

        # add rotary embedding
        return (
            add_rotary_embedding(q, cos, sin),
            add_rotary_embedding(k, cos, sin)
        )


