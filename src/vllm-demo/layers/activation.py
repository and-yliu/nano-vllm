import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# SwiGLU MLP:
# - gate_proj learns which features should pass and by how much. SiLU is applied to it.
# - up_proj learns the feature values/content that will pass through the gate. It normally has no activation before multiplication.
# - down_proj maps the expanded result back to the model’s hidden size

# gate = gate_proj(hidden_states)
# up   = up_proj(hidden_states)
# output = down_proj(F.silu(gate_proj(x)) * up_proj(x))


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # chunk(chunks, dim): splits a tensor into a specific number of smaller parts along a chosen axis
        x, y = x.chunk(2, -1)
        return F.silu(x) * y