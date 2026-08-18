#!/usr/bin/env python3
"""Same workload, three backends, one number each.

    python compare.py --backend mini --out results/mini.json
    python compare.py --backend hf   --out results/hf.json
    python compare.py --backend vllm --out results/vllm.json     # see note below
    python compare.py --compare results/*.json

One backend per invocation so each gets a clean process: vLLM and HF both
allocate aggressively, and a leftover KV pool from a previous backend would skew
whatever runs second. `pip install vllm` for the vllm backend.

Fairness rules enforced here:
  * identical prompts (same seed -> same random token ids, no tokenizer skew)
  * identical output length, with ignore_eos so no backend stops early
  * greedy decoding everywhere
  * same dtype (fp16 by default; bf16 is emulated on sm_75 and not comparable)
"""

import argparse
import json
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["mini", "hf", "vllm"])
    p.add_argument("--compare", nargs="+", metavar="JSON", help="print a table from result files")
    p.add_argument("--out", default=None, help="where to write the result json")

    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--num-seqs", type=int, default=64)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--prefix-len", type=int, default=0,
                   help="shared prefix across prompts; exercises prefix caching")

    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--hf-batch-size", type=int, default=32,
                   help="HF has no paging; a big batch reserves max_len KV per row and OOMs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# same as bench.py, generate randomized prompts
def make_prompts(args) -> list[list[int]]:
    """Random token ids, so results do not depend on any tokenizer's quirks."""
    rng = random.Random(args.seed)
    lo, hi = 100, 151_000          # inside Qwen3's vocab, off the special ids
    assert args.prefix_len <= args.input_len
    prefix = [rng.randint(lo, hi) for _ in range(args.prefix_len)]
    tail = args.input_len - args.prefix_len
    return [prefix + [rng.randint(lo, hi) for _ in range(tail)] for _ in range(args.num_seqs)]


# ---------------------------------------------------------------- backends
# run this project inference engine
def run_mini(args, prompts) -> dict:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from minivllm.engine.llm_engine import LLMEngine
    from minivllm.sampling_params import SamplingParams

    # build engine object
    engine = LLMEngine(
        args.model,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_length=args.input_len + args.output_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=getattr(torch, args.dtype),
    )
    params = SamplingParams(temperature=1e-6, max_tokens=args.output_len, ignore_eos=True)

    # add all generated prompts 
    for prompt in prompts:
        engine.add_request(prompt, params)

    # performance metrix as we step through engine
    prefill_s = decode_s = 0.0
    prefill_tokens = 0
    torch.cuda.synchronize()
    start = time.perf_counter()
    while engine.has_work():
        t = time.perf_counter()
        engine.step()
        dt = time.perf_counter() - t
        if engine.last_step_is_prefill:
            prefill_s += dt
            prefill_tokens += engine.last_step_num_tokens
        else:
            decode_s += dt
    torch.cuda.synchronize()
    wall = time.perf_counter() - start

    return {"wall_s": wall, "prefill_s": prefill_s, "decode_s": decode_s,
            "prompt_tokens_computed": prefill_tokens}

# run huggingface engine
def run_hf(args, prompts) -> dict:
    import torch
    from transformers import AutoModelForCausalLM
    from huggingface_hub import snapshot_download

    # load huggingface model
    path = snapshot_download(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=getattr(torch, args.dtype)
    ).cuda().eval()

    # measure performance
    torch.cuda.synchronize()
    start = time.perf_counter()
    # hf kv cache is [batch, kv_heads, seq, head_dim], need to control kv cache usage through hf_batch_size
    for i in range(0, len(prompts), args.hf_batch_size):
        chunk = prompts[i:i + args.hf_batch_size]
        input_ids = torch.tensor(chunk, dtype=torch.long, device="cuda")
        with torch.inference_mode():
            model.generate(
                input_ids,
                max_new_tokens=args.output_len,
                min_new_tokens=args.output_len,   # match ignore_eos
                do_sample=False,
                use_cache=True,
            )
    torch.cuda.synchronize()
    wall = time.perf_counter() - start

    return {"wall_s": wall, "prefill_s": None, "decode_s": None,
            "prompt_tokens_computed": args.num_seqs * args.input_len,
            "hf_batch_size": args.hf_batch_size}

# run vllm engine
def run_vllm(args, prompts) -> dict:
    import torch
    from vllm import LLM, SamplingParams as VLLMSamplingParams

    # create vllm model
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.input_len + args.output_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.prefix_len > 0,
    )
    params = VLLMSamplingParams(temperature=0.0, max_tokens=args.output_len, ignore_eos=True)

    # measure engine performace
    torch.cuda.synchronize()
    start = time.perf_counter()
    llm.generate(prompt_token_ids=prompts, sampling_params=params, use_tqdm=False)
    torch.cuda.synchronize()
    wall = time.perf_counter() - start

    return {"wall_s": wall, "prefill_s": None, "decode_s": None,
            "prompt_tokens_computed": args.num_seqs * args.input_len}


# ---------------------------------------------------------------- reporting

def gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"

# write one engine result to a json file
def report_one(args, result: dict) -> dict:
    out_tokens = args.num_seqs * args.output_len
    prompt_tokens = args.num_seqs * args.input_len
    record = {
        "backend": args.backend,
        "gpu": gpu_name(),
        "model": args.model,
        "dtype": args.dtype,
        "workload": {"num_seqs": args.num_seqs, "input_len": args.input_len,
                     "output_len": args.output_len, "prefix_len": args.prefix_len},
        "output_tokens": out_tokens,
        "prompt_tokens": prompt_tokens,
        "output_tok_s": out_tokens / result["wall_s"],
        "total_tok_s": (prompt_tokens + out_tokens) / result["wall_s"],
        **result,
    }
    print(f"\n{args.backend}  on {record['gpu']}")
    print(f"  workload      {args.num_seqs} x {args.input_len} in / {args.output_len} out"
          + (f", {args.prefix_len} shared prefix" if args.prefix_len else ""))
    print(f"  wall          {result['wall_s']:.2f}s")
    print(f"  output        {out_tokens:,} tok -> {record['output_tok_s']:,.0f} tok/s")
    print(f"  total         {record['total_tok_s']:,.0f} tok/s counting prompt")
    if result.get("prefill_s") is not None:
        print(f"  prefill/decode {result['prefill_s']:.2f}s / {result['decode_s']:.2f}s")
    if result.get("prompt_tokens_computed") is not None and args.prefix_len:
        served = prompt_tokens - result["prompt_tokens_computed"]
        if served > 0:
            print(f"  prefix cache  {served:,}/{prompt_tokens:,} prompt tokens served ({served/prompt_tokens:.0%})")
    return record

# compare results from different engine in the json file
def do_compare(paths: list[str]) -> None:
    records = [json.loads(Path(p).read_text()) for p in paths]
    records.sort(key=lambda r: -r["output_tok_s"])

    w = records[0]["workload"]
    print(f"\n{records[0]['model']} on {records[0]['gpu']}, {records[0]['dtype']}")
    print(f"{w['num_seqs']} seqs x {w['input_len']} in / {w['output_len']} out"
          + (f", {w['prefix_len']} shared prefix" if w["prefix_len"] else ""))
    if len({(r["workload"]["num_seqs"], r["workload"]["input_len"],
             r["workload"]["output_len"]) for r in records}) > 1:
        print("!! workloads differ between files -- not comparable")
    print("-" * 58)
    print(f"{'backend':<10} {'wall':>8} {'output tok/s':>14} {'vs slowest':>12}")
    print("-" * 58)
    base = min(r["output_tok_s"] for r in records)
    for r in records:
        print(f"{r['backend']:<10} {r['wall_s']:>7.2f}s {r['output_tok_s']:>14,.0f} "
              f"{r['output_tok_s'] / base:>11.2f}x")
    print("-" * 58)


def main() -> None:
    args = parse_args()

    if args.compare:
        do_compare(args.compare)
        return
    if not args.backend:
        raise SystemExit("pass --backend {mini,hf,vllm} or --compare FILES")

    prompts = make_prompts(args)
    runner = {"mini": run_mini, "hf": run_hf, "vllm": run_vllm}[args.backend]

    print(f"running backend={args.backend} ...")
    record = report_one(args, runner(args, prompts))

    out = args.out or f"results/{args.backend}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
