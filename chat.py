#!/usr/bin/env python3
"""Interactive chat against the engine.

    python chat.py
    python chat.py --temperature 0.7 --max-tokens 512

Multi-turn: the whole conversation is re-sent each turn, so every turn after the
first re-prefills the same leading tokens. That is exactly what the prefix cache
is for -- `/stats` reports how much of each prompt it served.

Commands:  /reset  clear history   /stats  cache + timing   /exit
"""

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from minivllm.engine.sequence import SequenceStatus
from minivllm.sampling_params import SamplingParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=16,
                   help="smaller blocks share prefixes at finer granularity")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--think", action="store_true",
                   help="let Qwen3 emit its <think> reasoning block")
    p.add_argument("--system", default=None, help="optional system prompt")
    return p.parse_args()


def build_prompt(tokenizer, messages: list[dict], think: bool) -> list[int]:
    # Render the conversation with the model's chat template -> as a flat list token ids.
    return tokenizer.apply_chat_template(
        messages, enable_thinking=think,
        add_generation_prompt=True, tokenize=True, return_dict=False,
    )


def stream_reply(engine, prompt_ids: list[int], params: SamplingParams) -> tuple[str, dict]:
    # Run one request to completion, printing tokens as they are produced.
    # add reuqest to engine
    seq_id = engine.add_request(prompt_ids, params)
    seq = engine.requests[seq_id]

    # performance variable
    shown = 0          # characters already printed
    prefill_s = 0.0
    decode_s = 0.0
    cached_tokens = 0
    first_token_s = None
    start = time.perf_counter()

    interrupted = False
    while engine.has_work():
        step_start = time.perf_counter()
        try:
            # run a step of the engine, do a prefill or decode batche
            finished = engine.step()
        except KeyboardInterrupt:
            # Retire the request by hand so the engine does not keep decoding it
            # forever; there is no abort() on LLMEngine yet.
            interrupted = True
            sched = engine.scheduler
            if seq in sched.running:
                sched.running.remove(seq)
            if seq in sched.waiting:
                sched.waiting.remove(seq)
            sched.block_manager.deallocate(seq)
            seq.status = SequenceStatus.FINISHED
            print("\n\033[2m  [interrupted]\033[0m")
            break
        elapsed = time.perf_counter() - step_start

        if engine.last_step_is_prefill:
            # add time
            prefill_s += elapsed
            # number of cache token hits, how many token is sent minus how many token the engine processed
            cached_tokens = len(prompt_ids) - engine.last_step_num_tokens
        else:
            decode_s += elapsed

        # get output tokens
        text = engine.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True)
        if len(text) > shown:
            # get time to generate first token 
            if first_token_s is None:
                first_token_s = time.perf_counter() - start
            # print text in stream, each step only generate the next token
            print(text[shown:], end="", flush=True)
            shown = len(text)

        # if finished stop loop
        if any(f.seq_id == seq_id for f in finished):
            break

    print()
    # return generate text and stats
    text = engine.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True)
    stats = {
        "prompt_tokens": len(prompt_ids),
        "cached_tokens": max(0, cached_tokens),
        "output_tokens": len(seq.output_token_ids),
        "ttft_s": first_token_s or 0.0,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
    }
    engine.requests.pop(seq_id, None)
    return text, stats


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("chat.py requires CUDA: attention is implemented as Triton kernels.")

    from minivllm.engine.llm_engine import LLMEngine

    print(f"loading {args.model} ...")
    t0 = time.perf_counter()
    # build engine
    engine = LLMEngine(
        args.model,
        block_size=args.block_size,
        max_model_length=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(f"ready in {time.perf_counter() - t0:.1f}s  "
          f"({engine.runner.num_blocks:,} KV blocks = "
          f"{engine.runner.num_blocks * args.block_size:,} tokens)")
    print("commands: /reset  /stats  /exit\n")

    # sampling parameters
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    totals = {"prompt_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
              "prefill_s": 0.0, "decode_s": 0.0, "turns": 0}

    while True:
        # get user inputs
        try:
            user = input("\033[1muser>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        # inputs of different command
        if not user:
            continue
        if user in ("/exit", "/quit"):
            # quit
            break
        if user == "/reset":
            # clear history
            messages = [m for m in messages if m["role"] == "system"]
            print("history cleared\n")
            continue
        if user == "/stats":
            # stats
            t = totals
            if t["turns"]:
                hit = t["cached_tokens"] / t["prompt_tokens"] if t["prompt_tokens"] else 0
                print(f"  turns          {t['turns']}")
                print(f"  prompt tokens  {t['prompt_tokens']:,} ({hit:.0%} served from prefix cache)")
                print(f"  output tokens  {t['output_tokens']:,}")
                if t["decode_s"]:
                    print(f"  decode         {t['output_tokens'] / t['decode_s']:,.1f} tok/s")
                if t["prefill_s"]:
                    print(f"  prefill        {t['prefill_s']:.2f}s total")
            else:
                print("  no turns yet")
            print()
            continue

        # add user message to history
        messages.append({"role": "user", "content": user})
        # build the prompt to input to engine
        prompt_ids = build_prompt(engine.tokenizer, messages, args.think)

        # check if the response will exceed the model's context
        if len(prompt_ids) + args.max_tokens > args.max_model_len:
            print(f"  conversation too long ({len(prompt_ids)} tokens); /reset or raise "
                  f"--max-model-len\n")
            messages.pop()
            continue

        print("\033[1massistant>\033[0m ", end="", flush=True)
        # get generated text and token through running the engine
        reply, stats = stream_reply(engine, prompt_ids, params)
        messages.append({"role": "assistant", "content": reply})

        # cumulate stats
        for k in ("prompt_tokens", "cached_tokens", "output_tokens", "prefill_s", "decode_s"):
            totals[k] += stats[k]
        totals["turns"] += 1

        # token generation speed (tok/s) in decode
        rate = stats["output_tokens"] / stats["decode_s"] if stats["decode_s"] else 0
        print(f"\033[2m  {stats['output_tokens']} tok, "
              f"ttft {stats['ttft_s'] * 1000:.0f}ms, {rate:,.1f} tok/s\033[0m\n")


if __name__ == "__main__":
    main()
