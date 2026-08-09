import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from vllm.utils.distributed import get_rank, get_world_size

class BaseLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = False, tp_dim: int | None = None ):
        super().__init__()
        # 0 for column, 1 for row
        self.tp_dim = tp_dim
        # the rank(numbering) of the gpu
        self.tp_rank = get_rank()
        # total number of gpu
        self.tp_size = get_world_size()

        self.weight = nn.Parameter(torch.empty((output_size, input_size)))
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)


    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        raise NotImplementedError


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
        tp_size = get_world_size()
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

class QKVMergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False
    ):
        self.tp_size = get_world_size()
        num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        # Number of heads for one gpu
        self.num_heads = num_heads // self.tp_size
        self.num_kv_heads = num_kv_heads // self.tp_size
        # the output size of one gpu
        self.output_size = self.head_size * (self.num_heads + 2 * self.num_kv_heads)
        total_output_size = head_size * (num_heads + 2 * num_kv_heads)

        super().__init__(input_size, total_output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_load_id: str):
        param_data = param.data

        assert weight_load_id in ['q', 'k', 'v']

        # offset: the starting position in locally stored param
        # shard_size: the length of the portion of param to copy
        if weight_load_id == "q":
            offset = 0
            shard_size = self.head_size * self.num_heads 
        elif weight_load_id == "k":
            offset = self.head_size * (self.num_heads)
            shard_size = self.head_size * self.num_kv_heads
        elif  weight_load_id == "v":
            offset = self.head_size * (self.num_heads + self.num_kv_heads)
            shard_size = self.head_size * self.num_kv_heads
        else:
            raise ValueError(f"Unknown weight_load_id: {weight_load_id}")

        param_data = param_data.narrow(self.tp_dim, offset, shard_size)

        # loaded_weight_start_index: the starting position in loaded weight(q weight/ k weight/ v weight)
        loaded_weight_start_index = shard_size * self.tp_rank

        # shard the original weights and load the portion to the portion in param
        loaded_weight = loaded_weight.narrow(self.tp_dim, loaded_weight_start_index, shard_size)
        param_data.copy_(loaded_weight)

class RowParallelLinear(BaseLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False
    ):
        self.tp_size = get_world_size()
        assert output_size % self.tp_size == 0
        super().__init__(input_size // self.tp_size, output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data

        # weight size per GPU
        shard_size = param_data.size(self.tp_dim)
        # start index of the loaded weights
        start_index = self.tp_rank * shard_size

        assert shard_size * self.tp_size == loaded_weight.size(self.tp_dim)

        loaded_weight = loaded_weight.narrow(self.tp_dim, start_index, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    
        

        
