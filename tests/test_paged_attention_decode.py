from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def write_sequence_to_cache(
    cache: torch.Tensor,
    values: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> None:
    for token_idx in range(values.size(0)):
        logical_block = token_idx // block_size
        block_offset = token_idx % block_size
        physical_block = int(block_table[logical_block].item())
        cache[physical_block, block_offset].copy_(values[token_idx])


def read_sequence_from_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    context_len: int,
    block_size: int,
) -> torch.Tensor:
    values = []
    for token_idx in range(context_len):
        logical_block = token_idx // block_size
        block_offset = token_idx % block_size
        physical_block = int(block_table[logical_block].item())
        values.append(cache[physical_block, block_offset])
    return torch.stack(values)


def reference_paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    num_q_head: int,
    num_kv_head: int,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    output = torch.empty_like(q)
    q_per_kv = num_q_head // num_kv_head

    for batch_idx in range(q.size(0)):
        context_len = int(context_lens[batch_idx].item())
        k_seq = read_sequence_from_cache(
            k_cache, block_tables[batch_idx], context_len, block_size
        )
        v_seq = read_sequence_from_cache(
            v_cache, block_tables[batch_idx], context_len, block_size
        )

        for q_head in range(num_q_head):
            kv_head = q_head // q_per_kv
            scores = q[batch_idx, q_head].float() @ k_seq[:, kv_head].float().T
            probs = torch.softmax(scores * scale, dim=-1)
            output[batch_idx, q_head] = (probs @ v_seq[:, kv_head].float()).to(q.dtype)

    return output


class PagedAttentionDecodeTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and Triton")
    def test_paged_attention_decode_matches_pytorch_reference(self) -> None:
        from vllm.layers.attention import paged_attention_decode

        torch.manual_seed(0)
        device = torch.device("cuda")
        dtype = torch.float16
        batch_size = 2
        num_q_head = 4
        num_kv_head = 2
        head_dim = 16
        block_size = 4
        scale = head_dim**-0.5

        context_lens = torch.tensor([5, 7], dtype=torch.int32, device=device)
        block_tables = torch.tensor(
            [
                [2, 0],
                [1, 3],
            ],
            dtype=torch.int32,
            device=device,
        )
        num_physical_blocks = 4

        q = torch.randn(batch_size, num_q_head, head_dim, device=device, dtype=dtype)
        k_cache = torch.zeros(
            num_physical_blocks,
            block_size,
            num_kv_head,
            head_dim,
            device=device,
            dtype=dtype,
        )
        v_cache = torch.zeros_like(k_cache)

        for batch_idx in range(batch_size):
            context_len = int(context_lens[batch_idx].item())
            k_seq = torch.randn(context_len, num_kv_head, head_dim, device=device, dtype=dtype)
            v_seq = torch.randn(context_len, num_kv_head, head_dim, device=device, dtype=dtype)
            write_sequence_to_cache(k_cache, k_seq, block_tables[batch_idx], block_size)
            write_sequence_to_cache(v_cache, v_seq, block_tables[batch_idx], block_size)

        actual = paged_attention_decode(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            num_q_head=num_q_head,
            num_kv_head=num_kv_head,
            head_dim=head_dim,
            block_size=block_size,
            scale=scale,
        )
        expected = reference_paged_attention_decode(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            num_q_head=num_q_head,
            num_kv_head=num_kv_head,
            block_size=block_size,
            scale=scale,
        )

        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
