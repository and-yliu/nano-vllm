import torch
import torch.nn as nn
import time

# RMSNorm: Root Mean Square Normalization
# RMS(x) = sqrt(sum(x^2)/len(x) + eps)
# RMSNorm(x) = x / RMS(x) ⊙ gamma
# RMSNorm keeps the activation magnitude within a more predictable range, which helps gradients flow through deep Transformer layers.
# The idea is to Keep the direction and relative feature values, but control the overall magnitude.
# Usually used in pre-norm

#LayerNorm performs both re-centering and re-scaling.
#RMSNorm only performs re-scaling.

class RMSNorm(nn.Module):
    def __init__(self, gamma: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        # nn.Parameter make gamma a learnable parameter
        self.weight = nn.Parameter(gamma.detach().clone())
        self.eps = eps

    @property
    def gamma(self):
        return self.weight

    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        rms = variance.sqrt()
        return x / rms * self.weight

    def residual_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor: 
        # residual help avoid diminishing gradient, perserve information
        x = x + residual
        return self.rms_forward(x)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is not None:
            return self.residual_rms_forward(x, residual)
        else:
            return self.rms_forward(x)

if __name__ == "__main__":
    x = torch.randn((4000, 8000)).cuda()
    gamma = torch.full((8000,), 0.5, dtype=x.dtype, device="cuda")

    layer = RMSNorm(gamma)
    residual = torch.full_like(x, fill_value=1)

    for _ in range(10):
        _ = layer(x)

    # without residual
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.time()
        _ = layer(x)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    print(f"Without Residual: {avg_time * 1000:.3f} ms")

    # with residual
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.time()
        _ = layer(x, residual)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    print(f"With Residual: {avg_time * 1000:.3f} ms")





