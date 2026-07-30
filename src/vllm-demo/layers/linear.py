import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

class BaseLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = False, tp_dim: int | None = None ):
        super().__init__()
        # 0 for column, 1 for row
        self.tp_dim = tp_dim
        # the rank(numbering) of the gpu
        self.tp_rank = dist.get_rank()
        # total number of gpu
        self.tp_size = dist.get_world_size()

        self.weight = nn.Parameter(torch.empty((output_size, input_size)))
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)


    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        return NotImplementedError

    def forward(self, x: torch.Tensor):
        return NotImplementedError


# most simple linear transformation
class ReplicatedLinear(BaseLinear):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(BaseLinear):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = dist.get_world_size()
        super().__init__(input_size, output_size // tp_size, bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_index = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_index, shard_size)
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight, self.bias)

class MergeColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self, 
        input_size: int, 
        output_size: list[int],  # merge multiple linear transformation and then split
        bias: bool = False
    ):
        self.output_size = output_size
        super().__init__(input_size, sum(output_size), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_load_id: int):
        param_data = param.data

        # offset is where to load weights in param.data
        offset = sum(self.output_size[:weight_load_id]) // self.tp_size
        # how many data to load starting from offset
        shard_size = self.output_size[weight_load_id] // self.tp_size
        # which section of param_data to load weight
        param_data = param_data.narrow(self.tp_dim, offset, shard_size)

        # getting the correct portion of loaded_weights
        loaded_weight_start_index = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, loaded_weight_start_index, shard_size)
        # puts it into the correct parameter section
        param_data.copy_(loaded_weight)
        
