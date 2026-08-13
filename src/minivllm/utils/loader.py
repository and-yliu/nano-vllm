from glob import glob

import torch
import torch.nn as nn
from safetensors import safe_open

def default_load_weight(param: nn.Parameter, weights: torch.Tensor):
    param.data.copy_(weights)


def load_model(model: nn.Module, path: str):
    """Load a HuggingFace safetensors checkpoint into `model` in place.

    Module names deliberately mirror the HuggingFace checkpoint, so no name
    translation is needed -- only `packed_modules_mapping`, which encodes the
    fused parameters (qkv_linear, gate_up) that have no 1:1 checkpoint tensor.

    Iterates the checkpoint rather than the model's parameters, so that those
    fused parameters can receive their shards one at a time.

    Raises if any checkpoint key is unrecognised, or if any model parameter is
    never written -- a silently unloaded weight is the worst failure mode here,
    since it surfaces only as subtly wrong logits much later.
    """

    # get all params of my model
    params = dict(model.named_parameters())
    # (param name, shard id) pairs, so a fused param that received only 2 of its
    # 3 shards is still detected as incomplete.
    seen: set[tuple[str, object]] = set()

    # load the safetensor file to get params from checkpoint
    shard_files = sorted(glob(f"{path}/*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"no *.safetensors found under {path}")

    # loops through all the files
    for shard_file in shard_files:
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            # for each checkpoint tensor, see which model param does it belong to
            for hf_name in f.keys():
                name = hf_name
                shard_id = None

                # for fused params, need shard_id to load
                for suffix, (target, sid) in model.packed_modules_mapping.items():
                    if suffix in name:
                        name, shard_id = name.replace(suffix, target), sid
                        break

                # if not found, raise an error
                if name not in params:
                    raise KeyError(
                        f"checkpoint key {hf_name!r} mapped to {name!r}, which is "
                        f"not a parameter of {type(model).__name__}"
                    )

                # get param object, loaded_weights tensor from file, and weight loader of the param
                param = params[name]
                tensor = f.get_tensor(hf_name)
                weight_loader = getattr(param, "weight_loader", None)

                # load the params throught weight loader
                if weight_loader is None:
                    default_load_weight(param, tensor)
                elif shard_id is None:
                    weight_loader(param, tensor)
                else:
                    weight_loader(param, tensor, shard_id)

                # add to seen
                seen.add((name, shard_id))

    # check if there are missing params for the model
    missing = _expected_shards(model, params) - seen
    if missing:
        raise RuntimeError(
            f"{len(missing)} parameter shard(s) never loaded, e.g. "
            f"{sorted(missing, key=str)[:5]}"
        )

# get what params are expected to load to the model
def _expected_shards(model: nn.Module, params: dict) -> set:
    """Every (param name, shard id) pair a complete checkpoint should produce."""
    fused: dict[str, list] = {}
    for target, sid in model.packed_modules_mapping.values():
        fused.setdefault(target, []).append(sid)

    expected = set()
    for name in params:
        # "model.layers.0.self_attn.qkv_linear.weight" -> "qkv_linear"
        parts = name.split(".")
        module = parts[-2] if len(parts) >= 2 else ""
        if module in fused:
            expected.update((name, sid) for sid in fused[module])
        else:
            expected.add((name, None))
    return expected
