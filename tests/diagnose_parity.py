import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_PROMPT = "The capital of France is"


def build(path, device, dtype):
    from transformers import AutoModelForCausalLM, Qwen3Config

    from minivllm.model.qwen3 import Qwen3ForCausalLM
    from minivllm.utils.loader import load_model

    ref = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()

    config = Qwen3Config.from_pretrained(path)
    ours = Qwen3ForCausalLM(config).to(device=device, dtype=dtype)
    load_model(ours, path)
    return ours.eval(), ref


def our_logits(model, token_ids, device):
    from minivllm.utils.context import reset_context, set_context

    cu = torch.tensor([0, token_ids.numel()], dtype=torch.int32, device=device)
    set_context(is_prefill=True, cu_seqlens_q=cu)
    try:
        with torch.inference_mode():
            return model.compute_logits(model(token_ids))[0].float()
    finally:
        reset_context()


def ref_logits(ref, token_ids):
    with torch.inference_mode():
        return ref(token_ids.unsqueeze(0)).logits[0, -1].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("requires CUDA")

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    path = snapshot_download(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(path)

    ours, ref = build(path, device, dtype)
    prompt_ids = tok(args.prompt, return_tensors="pt").input_ids[0].to(device)

    # Reference's own greedy continuation -- the fixed context both models see.
    with torch.inference_mode():
        seq = ref.generate(prompt_ids.unsqueeze(0), max_new_tokens=args.steps,
                           do_sample=False, use_cache=True,
                           pad_token_id=tok.eos_token_id)[0]

    print(f"dtype={args.dtype}  prompt={args.prompt!r}")
    print(f"reference continuation: {tok.decode(seq[prompt_ids.numel():])!r}\n")
    print(f"{'step':>4} {'agree':>6} {'top2 gap':>9} {'max diff':>9}  ours / ref")
    print("-" * 72)

    disagreements = []
    for t in range(prompt_ids.numel(), seq.numel()):
        prefix = seq[:t]
        a, b = our_logits(ours, prefix, device), ref_logits(ref, prefix)

        top2 = torch.topk(b, 2).values
        gap = (top2[0] - top2[1]).item()
        max_diff = (a - b).abs().max().item()
        ours_tok, ref_tok = a.argmax().item(), b.argmax().item()
        agree = ours_tok == ref_tok

        if not agree:
            disagreements.append((t, gap, max_diff))

        note = "" if agree else f"  {tok.decode([ours_tok])!r} / {tok.decode([ref_tok])!r}"
        print(f"{t:>4} {str(agree):>6} {gap:>9.4f} {max_diff:>9.4f}{note}")

    print("-" * 72)
    if not disagreements:
        print("No disagreement on any identical prefix -- implementations match.")
        print("Free-running greedy divergence is pure numerical drift.")
    else:
        print(f"{len(disagreements)} disagreement(s):")
        for t, gap, md in disagreements:
            verdict = "near-tie (numerics)" if gap < 0.1 else "LARGE GAP -- suspect a real bug"
            print(f"  step {t}: top2 gap {gap:.4f}, max logit diff {md:.4f} -> {verdict}")


if __name__ == "__main__":
    main()
