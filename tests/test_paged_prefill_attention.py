"""Paged prefill: attention over a KV cache, so a cached prefix can be skipped.

The property that matters is chunked-prefill equivalence -- prefilling a
sequence in one pass must produce the same thing as prefilling part of it and
then running the rest against the cached prefix. That is the whole contract
prefix caching relies on.

    python -m unittest tests.test_paged_prefill_attention -v
"""

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def reference_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_q_head: int,
    num_kv_head: int,
    scale: float,
) -> torch.Tensor:
    """Plain causal attention over one contiguous sequence, in fp32."""
    seq_len = q.shape[0]
    q_per_kv = num_q_head // num_kv_head
    output = torch.empty_like(q)
    causal_mask = torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device)
    )

    for q_head in range(num_q_head):
        kv_head = q_head // q_per_kv
        scores = q[:, q_head].float() @ k[:, kv_head].float().T
        scores = scores * scale
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        output[:, q_head] = (probs @ v[:, kv_head].float()).to(q.dtype)

    return output


def scatter_into_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: list[int],
    block_size: int,
) -> None:
    """Write one sequence's K/V into the paged cache, honouring its block table."""
    for token_index in range(k.shape[0]):
        physical = block_table[token_index // block_size]
        offset = token_index % block_size
        k_cache[physical, offset] = k[token_index]
        v_cache[physical, offset] = v[token_index]


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and Triton")
class PagedPrefillAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self.num_q_head = 4
        self.num_kv_head = 2
        self.head_dim = 32
        self.block_size = 16
        self.scale = self.head_dim**-0.5

    def _make_cache(self, num_blocks: int):
        shape = (num_blocks, self.block_size, self.num_kv_head, self.head_dim)
        return (
            torch.zeros(shape, device=self.device, dtype=self.dtype),
            torch.zeros(shape, device=self.device, dtype=self.dtype),
        )

    def _random_qkv(self, num_tokens: int):
        q = torch.randn(
            num_tokens, self.num_q_head, self.head_dim, device=self.device, dtype=self.dtype
        )
        k = torch.randn(
            num_tokens, self.num_kv_head, self.head_dim, device=self.device, dtype=self.dtype
        )
        v = torch.randn(
            num_tokens, self.num_kv_head, self.head_dim, device=self.device, dtype=self.dtype
        )
        return q, k, v

    def test_no_cached_prefix_matches_reference(self) -> None:
        """num_cached == 0: same maths as the contiguous kernel, via the cache."""
        from minivllm.layers.attention import paged_prefill_attention

        seq_len = 40
        q, k, v = self._random_qkv(seq_len)

        block_table = [0, 1, 2]
        k_cache, v_cache = self._make_cache(num_blocks=4)
        scatter_into_cache(k, v, k_cache, v_cache, block_table, self.block_size)

        actual = paged_prefill_attention(
            q, k_cache, v_cache,
            cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32, device=self.device),
            cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=self.device),
            block_tables=torch.tensor([block_table], dtype=torch.int32, device=self.device),
            num_q_head=self.num_q_head,
            num_kv_head=self.num_kv_head,
            head_dim=self.head_dim,
            block_size=self.block_size,
            scale=self.scale,
        )
        expected = reference_causal_attention(
            q, k, v, self.num_q_head, self.num_kv_head, self.scale
        )

        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_chunked_prefill_equivalence(self) -> None:
        """Prefilling 32+8 must equal prefilling 40 in one pass.

        This is the assertion prefix caching stands on: if it holds, skipping a
        cached prefix is free of consequence.
        """
        from minivllm.layers.attention import paged_prefill_attention

        seq_len, num_cached = 40, 32
        q, k, v = self._random_qkv(seq_len)

        block_table = [0, 1, 2]
        k_cache, v_cache = self._make_cache(num_blocks=4)
        # the cached prefix AND this pass's own K/V are both in the cache by the
        # time attention runs -- save_kv_cache writes before the kernel is called
        scatter_into_cache(k, v, k_cache, v_cache, block_table, self.block_size)

        actual = paged_prefill_attention(
            q[num_cached:], k_cache, v_cache,
            cu_seqlens_q=torch.tensor(
                [0, seq_len - num_cached], dtype=torch.int32, device=self.device
            ),
            cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=self.device),
            block_tables=torch.tensor([block_table], dtype=torch.int32, device=self.device),
            num_q_head=self.num_q_head,
            num_kv_head=self.num_kv_head,
            head_dim=self.head_dim,
            block_size=self.block_size,
            scale=self.scale,
        )
        expected = reference_causal_attention(
            q, k, v, self.num_q_head, self.num_kv_head, self.scale
        )[num_cached:]

        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_varlen_batch_with_different_cached_lengths(self) -> None:
        """Two sequences, different context and cached lengths, one launch."""
        from minivllm.layers.attention import paged_prefill_attention

        # (total length, already cached)
        specs = [(40, 32), (20, 0)]
        block_tables = [[0, 1, 2], [3, 4, -1]]

        k_cache, v_cache = self._make_cache(num_blocks=6)
        per_seq = []
        for (seq_len, _), block_table in zip(specs, block_tables):
            q, k, v = self._random_qkv(seq_len)
            scatter_into_cache(k, v, k_cache, v_cache, block_table, self.block_size)
            per_seq.append((q, k, v))

        q_packed = torch.cat([q[cached:] for (q, _, _), (_, cached) in zip(per_seq, specs)])

        cu_q, cu_k = [0], [0]
        for seq_len, cached in specs:
            cu_q.append(cu_q[-1] + seq_len - cached)
            cu_k.append(cu_k[-1] + seq_len)

        actual = paged_prefill_attention(
            q_packed, k_cache, v_cache,
            cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=self.device),
            cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, device=self.device),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, device=self.device),
            num_q_head=self.num_q_head,
            num_kv_head=self.num_kv_head,
            head_dim=self.head_dim,
            block_size=self.block_size,
            scale=self.scale,
        )

        expected = torch.cat([
            reference_causal_attention(
                q, k, v, self.num_q_head, self.num_kv_head, self.scale
            )[cached:]
            for (q, k, v), (_, cached) in zip(per_seq, specs)
        ])

        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
