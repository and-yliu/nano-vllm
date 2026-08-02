import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# The classic FFN is FFN(x)=W2 @ ​ReLU(W1 @ ​x)
# This one uses a SwiGLU MLP, and the FFN is FFN(x)=[SiLU(x @ Wg​) ⊙ (x @ Wv​)] @ Wo​

# SwiGLU MLP (Multilayer Perception):
# - gate_proj learns which features should pass and by how much. SiLU is applied to it.
# - up_proj learns the feature values/content that will pass through the gate. It normally has no activation before multiplication.
# - down_proj maps the expanded result back to the model’s hidden size

# gate = gate_proj(hidden_states)
# up   = up_proj(hidden_states)
# output = down_proj(F.silu(gate_proj(x)) * up_proj(x))

# Advantages: 
# - More expressive than ReLU or GELU. SwiGLU produces two separate projections and lets one control the other. More flexibility in deciding which information matters for each input
# - Smooth gradients
# - Better performance at similar compute
# Disadvantages:
# - SwiGLU can require more parameters and matrix multiplication if the intermediate dimension remains unchanged. Usally smaller intermediate dimension


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # chunk(chunks, dim): splits a tensor into a specific number of smaller parts along a chosen axis
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

if __name__ == "__main__":
    layer = SiluAndMul().cuda()
    input_tensor = torch.randn(4000, 8000).cuda()

    for _ in range(10):
        _ = layer(input_tensor)

    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        cur = time.time()
        output_tensor = layer(input_tensor)
        torch.cuda.synchronize()
        times.append(cur - time.time())

    avg_time = sum(times) / len(times)
    print(f"Average compliation time = {avg_time * 1000:.3f} ms")




