#!/usr/bin/env python3
from pathlib import Path
import sys

import torch
from transformers import Qwen3Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minivllm.model.qwen3 import Qwen3ForCausalLM
from minivllm.utils.context import reset_context, set_context


def init_debug_weights(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if "norm" in name and param.ndim == 1:
            torch.nn.init.ones_(param)
        elif param.ndim >= 2:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
        else:
            torch.nn.init.zeros_(param)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("smoke_qwen3.py requires CUDA because attention uses Triton kernels.")

    device = torch.device("cuda")
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=128,
        attention_bias=False,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )

    model = Qwen3ForCausalLM(config).to(device=device, dtype=torch.float16)
    init_debug_weights(model)
    model.eval()

    token_ids = torch.tensor([1, 8, 12, 34, 55, 2, 9, 10], device=device)
    cu_seqlens_q = torch.tensor([0, token_ids.numel()], dtype=torch.int32, device=device)

    reset_context()
    set_context(is_prefill=True, cu_seqlens_q=cu_seqlens_q)

    with torch.inference_mode():
        hidden_states = model(token_ids)
        logits = model.compute_logits(hidden_states)

    print(f"hidden_states: {tuple(hidden_states.shape)}")
    print(f"logits: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
