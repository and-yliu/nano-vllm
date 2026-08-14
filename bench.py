#!/usr/bin/env python3
"""Throughput benchmark for the engine. Requires CUDA -- attention is Triton.

    python bench.py --model Qwen/Qwen3-0.6B --num-seqs 64 --input-len 512 --output-len 128

Prompts are random token ids, so nothing hits the prefix cache and the numbers
describe cold-start work. `--prefix-len N` prepends one shared random prefix to
every prompt instead, which is what exercises prefix caching -- compare the two
runs to see what the cache is worth.

Output lengths are fixed with ignore_eos so a run is reproducible: the model
cannot end a sequence early and change the amount of work between runs.

Prefill and decode are timed separately. They are different regimes -- prefill
is compute bound on big matmuls, decode is memory/launch bound at batch size
`num_seqs` -- and a single blended tok/s number hides which one a change moved.
"""

import argparse
import random
import time
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from minivllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from minivllm.engine.llm_engine import LLMEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF repo id or local path")
    p.add_argument("--num-seqs", type=int, default=64, help="requests in the run")
    p.add_argument("--input-len", type=int, default=512, help="prompt tokens per request")
    p.add_argument("--output-len", type=int, default=128, help="tokens to generate per request")
    p.add_argument("--prefix-len", type=int, default=0,
                   help="tokens of shared prefix across all prompts (counts toward --input-len)")

    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--max-num-seqs", type=int, default=256, help="batch width cap")
    p.add_argument("--max-num-batched-tokens", type=int, default=8192, help="prefill tokens per step")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=None,
                   help="defaults to input+output; keep it tight, warmup allocates this much")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", action="store_true", help="discard one short run before measuring")
    return p.parse_args()


def make_prompts(args: argparse.Namespace, vocab_size: int) -> list[list[int]]:
    rng = random.Random(args.seed)
    # Stay off the special-token ids at the bottom of the vocab; they are
    # untypical inputs and some are reserved.
    lo, hi = 100, vocab_size - 1

    assert args.prefix_len <= args.input_len, "--prefix-len exceeds --input-len"
    prefix = [rng.randint(lo, hi) for _ in range(args.prefix_len)]
    suffix_len = args.input_len - args.prefix_len
    return [prefix + [rng.randint(lo, hi) for _ in range(suffix_len)] for _ in range(args.num_seqs)]


def run(engine: "LLMEngine", prompts: list[list[int]], output_len: int) -> dict:
    params = SamplingParams(max_tokens=output_len, ignore_eos=True)
    for prompt in prompts:
        engine.add_request(prompt, params)

    stats = {"prefill_s": 0.0, "decode_s": 0.0, "prefill_tokens": 0,
             "decode_tokens": 0, "prefill_steps": 0, "decode_steps": 0}

    torch.cuda.synchronize()
    start = time.perf_counter()
    while engine.has_work():
        step_start = time.perf_counter()
        engine.step()
        # step() ends in a device->host copy of the sampled ids, so the GPU work
        # is already complete here; no synchronize needed per step.
        elapsed = time.perf_counter() - step_start

        kind = "prefill" if engine.last_step_is_prefill else "decode"
        stats[f"{kind}_s"] += elapsed
        stats[f"{kind}_tokens"] += engine.last_step_num_tokens
        stats[f"{kind}_steps"] += 1
    torch.cuda.synchronize()

    stats["wall_s"] = time.perf_counter() - start
    return stats


def report(args: argparse.Namespace, stats: dict) -> None:
    prompt_tokens = args.num_seqs * args.input_len
    output_tokens = args.num_seqs * args.output_len
    # The first token of each sequence comes out of the prefill step, so decode
    # steps produce output_len - 1 each.
    cached = prompt_tokens - stats["prefill_tokens"]

    def rate(n: int, seconds: float) -> str:
        return f"{n / seconds:,.0f} tok/s" if seconds > 0 else "n/a"

    print()
    print(f"model            {args.model}")
    print(f"workload         {args.num_seqs} seqs x {args.input_len} in / {args.output_len} out"
          + (f", {args.prefix_len} shared prefix" if args.prefix_len else ""))
    print(f"limits           block_size={args.block_size} max_num_seqs={args.max_num_seqs} "
          f"max_num_batched_tokens={args.max_num_batched_tokens}")
    print("-" * 72)
    print(f"prefill          {stats['prefill_tokens']:>9,} tok in {stats['prefill_s']:6.2f}s  "
          f"{rate(stats['prefill_tokens'], stats['prefill_s']):>14}   ({stats['prefill_steps']} steps)")
    print(f"decode           {stats['decode_tokens']:>9,} tok in {stats['decode_s']:6.2f}s  "
          f"{rate(stats['decode_tokens'], stats['decode_s']):>14}   ({stats['decode_steps']} steps)")
    if stats["decode_steps"]:
        print(f"decode step      {stats['decode_s'] / stats['decode_steps'] * 1000:.2f} ms mean "
              f"({stats['decode_tokens'] / stats['decode_steps']:.1f} seqs/step)")
    if cached > 0:
        print(f"prefix cache     {cached:,} of {prompt_tokens:,} prompt tokens served from cache "
              f"({cached / prompt_tokens:.0%})")
    elif cached < 0:
        # Preempted sequences are re-prefilled from scratch, so more prompt
        # tokens were computed than exist. Means the KV cache is too small for
        # this batch: lower --num-seqs or raise --gpu-memory-utilization.
        print(f"preemption       {-cached:,} extra prompt tokens recomputed "
              f"({-cached / prompt_tokens:.0%} over) -- KV cache is oversubscribed")
    print("-" * 72)
    print(f"total            {stats['wall_s']:.2f}s wall, "
          f"{output_tokens / stats['wall_s']:,.0f} output tok/s, "
          f"{(stats['prefill_tokens'] + stats['decode_tokens']) / stats['wall_s']:,.0f} tok/s counting prefill")
    print()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("bench.py requires CUDA: attention is implemented as Triton kernels.")

    from minivllm.engine.llm_engine import LLMEngine

    torch.manual_seed(args.seed)
    # Warmup prefills max_model_len tokens in one pass and that peak is what
    # sizes the KV cache, so leaving this at the model default (40k for Qwen3)
    # both wastes startup time and shrinks the cache.
    max_model_len = args.max_model_len or (args.input_len + args.output_len)

    print(f"loading {args.model} ...")
    load_start = time.perf_counter()
    engine = LLMEngine(
        args.model,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_length=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(f"ready in {time.perf_counter() - load_start:.1f}s, "
          f"{engine.runner.num_blocks:,} KV blocks "
          f"({engine.runner.num_blocks * args.block_size:,} tokens)")

    prompts = make_prompts(args, engine.runner.config.vocab_size)

    if args.warmup:
        rng = random.Random(args.seed + 1)
        vocab = engine.runner.config.vocab_size
        run(engine, [[rng.randint(100, vocab - 1) for _ in range(64)] for _ in range(2)], output_len=4)

    report(args, run(engine, prompts, args.output_len))


if __name__ == "__main__":
    main()
