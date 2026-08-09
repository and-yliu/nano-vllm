"""End-to-end parity against HuggingFace on real Qwen3-0.6B weights.

This is the ground-truth test for the loader and the prefill path: it asserts
that this implementation produces the same logits, and the same greedy tokens,
as `transformers` on the same prompt.

Requires CUDA (the attention kernels are Triton) and downloads ~1.5GB on first
run into the HuggingFace cache.

    python -m unittest tests.test_parity -v
"""

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"
N_GREEDY_TOKENS = 20

# bf16 accumulates differently across two independent attention implementations,
# so logits are compared loosely; the argmax and the greedy token stream are the
# assertions that actually pin down correctness.
LOGIT_ATOL = 0.35


def _cuda_available() -> bool:
    return torch.cuda.is_available()


@unittest.skipUnless(_cuda_available(), "parity test requires CUDA")
class TestQwen3Parity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

        from vllm.model.qwen3 import Qwen3ForCausalLM
        from vllm.utils.loader import load_model

        cls.path = snapshot_download(MODEL_ID)
        cls.device = torch.device("cuda")
        # bf16 needs Ampere (sm_80+). On older cards (T4 = sm_75) it is emulated,
        # slower, and less accurate than fp16, so prefer fp16 there.
        major, _ = torch.cuda.get_device_capability()
        cls.dtype = torch.bfloat16 if major >= 8 else torch.float16

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.path)
        cls.config = Qwen3Config.from_pretrained(cls.path)

        cls.ref = AutoModelForCausalLM.from_pretrained(
            cls.path, dtype=cls.dtype
        ).to(cls.device).eval()

        model = Qwen3ForCausalLM(cls.config).to(device=cls.device, dtype=cls.dtype)
        load_model(model, cls.path)
        cls.model = model.eval()

        cls.token_ids = cls.tokenizer(PROMPT, return_tensors="pt").input_ids[0].to(cls.device)

    @classmethod
    def tearDownClass(cls):
        cls.ref = cls.model = None
        torch.cuda.empty_cache()

    # -- helpers ---------------------------------------------------------

    def _ours_last_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Logits for the final token of a single prompt: [vocab]."""
        from vllm.utils.context import reset_context, set_context

        cu_seqlens_q = torch.tensor(
            [0, token_ids.numel()], dtype=torch.int32, device=self.device
        )
        set_context(is_prefill=True, cu_seqlens_q=cu_seqlens_q)
        try:
            with torch.inference_mode():
                hidden = self.model(token_ids)
                logits = self.model.compute_logits(hidden)
        finally:
            reset_context()
        return logits[0].float()

    def _ref_last_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            out = self.ref(token_ids.unsqueeze(0))
        return out.logits[0, -1].float()

    # -- tests -----------------------------------------------------------

    def test_all_parameters_loaded(self):
        """load_model raises on gaps, so reaching here means every shard landed.

        Additionally check nothing kept its constructor init: an unwritten
        RMSNorm stays all-ones, which is easy to miss.
        """
        for name, param in self.model.named_parameters():
            self.assertFalse(
                torch.isnan(param).any(), f"{name} contains NaN",
            )
            if "norm" in name:
                self.assertFalse(
                    torch.all(param == 1.0),
                    f"{name} is still all-ones -- it was never loaded",
                )

    def test_last_token_logits_match(self):
        ours = self._ours_last_logits(self.token_ids)
        ref = self._ref_last_logits(self.token_ids)

        self.assertEqual(ours.shape, ref.shape)

        max_diff = (ours - ref).abs().max().item()
        self.assertEqual(
            ours.argmax().item(),
            ref.argmax().item(),
            f"argmax disagrees (max abs logit diff {max_diff:.4f}); "
            f"{self._divergence_report()}",
        )
        self.assertLess(
            max_diff,
            LOGIT_ATOL,
            f"max abs logit diff {max_diff:.4f} exceeds {LOGIT_ATOL}; "
            f"{self._divergence_report()}",
        )

    def test_top5_predictions_match(self):
        ours = self._ours_last_logits(self.token_ids).topk(5).indices.tolist()
        ref = self._ref_last_logits(self.token_ids).topk(5).indices.tolist()
        self.assertEqual(ours, ref, "top-5 next-token predictions disagree")

    def test_greedy_generation_matches(self):
        """Naive greedy loop -- re-prefills every step, no KV cache.

        Slow and quadratic, but it is the oracle the paged/batched engine will
        later have to reproduce exactly.
        """
        ours = self.token_ids.clone()
        for _ in range(N_GREEDY_TOKENS):
            nxt = self._ours_last_logits(ours).argmax()
            ours = torch.cat([ours, nxt.reshape(1)])

        with torch.inference_mode():
            ref = self.ref.generate(
                self.token_ids.unsqueeze(0),
                max_new_tokens=N_GREEDY_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]

        ours_new = ours[self.token_ids.numel():].tolist()
        ref_new = ref[self.token_ids.numel():].tolist()

        self.assertEqual(
            ours_new,
            ref_new,
            "greedy token streams diverge\n"
            f"  ours: {self.tokenizer.decode(ours_new)!r}\n"
            f"  ref:  {self.tokenizer.decode(ref_new)!r}",
        )

    # -- debugging -------------------------------------------------------

    def _divergence_report(self) -> str:
        """Find the first decoder layer whose output differs from the reference.

        Called only when an assertion has already failed, to point at the
        offending layer instead of leaving a bare logit diff.
        """
        from vllm.utils.context import reset_context, set_context

        ours_out, ref_out = {}, {}

        def grab(store, idx):
            def hook(_module, _inputs, output):
                t = output[0] if isinstance(output, tuple) else output
                store[idx] = t.detach().float()
            return hook

        handles = []
        for i, layer in enumerate(self.model.model.layers):
            handles.append(layer.register_forward_hook(grab(ours_out, i)))
        for i, layer in enumerate(self.ref.model.layers):
            handles.append(layer.register_forward_hook(grab(ref_out, i)))

        try:
            cu = torch.tensor([0, self.token_ids.numel()], dtype=torch.int32, device=self.device)
            set_context(is_prefill=True, cu_seqlens_q=cu)
            with torch.inference_mode():
                self.model(self.token_ids)
            reset_context()
            with torch.inference_mode():
                self.ref(self.token_ids.unsqueeze(0))
        finally:
            for h in handles:
                h.remove()

        for i in sorted(ours_out):
            if i not in ref_out:
                break
            a, b = ours_out[i], ref_out[i].reshape(ours_out[i].shape)
            diff = (a - b).abs().max().item()
            if diff > LOGIT_ATOL:
                return f"first divergence at decoder layer {i} (max abs diff {diff:.4f})"
        return "no single layer diverged past threshold -- suspect lm_head or final norm"


if __name__ == "__main__":
    unittest.main()
