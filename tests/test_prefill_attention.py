from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def reference_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_q_head: int,
    num_kv_head: int,
    scale: float,
) -> torch.Tensor:
    output = torch.empty_like(q)
    cu_seqlens_cpu = cu_seqlens.cpu().tolist()
    q_per_kv = num_q_head // num_kv_head

    for seq_idx in range(len(cu_seqlens_cpu) - 1):
        start = cu_seqlens_cpu[seq_idx]
        end = cu_seqlens_cpu[seq_idx + 1]
        seq_len = end - start
        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device)
        )

        for q_head in range(num_q_head):
            kv_head = q_head // q_per_kv
            scores = q[start:end, q_head].float() @ k[start:end, kv_head].float().T
            scores = scores * scale
            scores = scores.masked_fill(~causal_mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            output[start:end, q_head] = (probs @ v[start:end, kv_head].float()).to(q.dtype)

    return output


class PrefillAttentionTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and Triton")
    def test_prefill_flash_attention_matches_pytorch_reference(self) -> None:
        from minivllm.layers.attention import prefill_flash_attention

        torch.manual_seed(0)
        device = torch.device("cuda")
        dtype = torch.float16
        cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32, device=device)
        total_tokens = int(cu_seqlens[-1].item())
        num_q_head = 4
        num_kv_head = 2
        head_dim = 16
        scale = head_dim**-0.5

        q = torch.randn(total_tokens, num_q_head, head_dim, device=device, dtype=dtype)
        k = torch.randn(total_tokens, num_kv_head, head_dim, device=device, dtype=dtype)
        v = torch.randn(total_tokens, num_kv_head, head_dim, device=device, dtype=dtype)

        actual = prefill_flash_attention(
            q,
            k,
            v,
            cu_seqlens,
            num_q_head=num_q_head,
            num_kv_head=num_kv_head,
            head_dim=head_dim,
            scale=scale,
        )
        expected = reference_prefill_attention(
            q,
            k,
            v,
            cu_seqlens,
            num_q_head=num_q_head,
            num_kv_head=num_kv_head,
            scale=scale,
        )

        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
