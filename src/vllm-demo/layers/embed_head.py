import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embedding: int, embedding_dim: int):
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()

        # length of the embedding 
        self.embedding_dim = embedding_dim
        # size of the vocabulary
        self.num_embedding = num_embedding
        # pad the vocabulary so it is divisible by tp_size
        padding_num_embedding = (num_embedding + self.tp_size - 1) // self.tp_size * self.tp_size
        # number of embedding per gpu
        self.num_embedding_per_partition = padding_num_embedding // self.tp_size

        self.weight = nn.Parameter(torch.empty((self.num_embedding_per_partition, self.embedding_dim)))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, params: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = params.data

        offset = self.tp_rank * self.num_embedding_per_partition
        shard_size = self.num_embedding_per_partition

        # making sure that the range does not consider padding
        actual_start = min(offset, self.num_embedding)
        actual_end = min(offset + shard_size, self.num_embedding)
        actual_shard_size = min(0, actual_end - actual_start)

        if actual_shard_size > 0:
            loaded_weight = loaded_weight.narrow(0, actual_start, actual_shard_size)
            param_data[:actual_shard_size].copy_(loaded_weight)

        # add padding to the param
        if actual_shard_size < shard_size:
            param_data[actual_shard_size:].zero_()

    def forward(self, x: torch.Tensor):
        # determine which token are in this gpu
        mask = (x >= self.tp_rank * self.num_embedding_per_partition) & (x < (self.tp_rank+1) * self.num_embedding_per_partition) & (x < self.num_embedding)
        # set token index to relative index for this gpu (but 0 stand for both not in this gpu and relative index 0)
        x = mask * (x - self.tp_rank * self.num_embedding_per_partition)

        # get embedding from the weight(embedding table)
        output = F.embedding(x, self.weight)

        if self.tp_size > 1:
            # clear values for token not in this gpu
            output = output * mask.unsqueeze_(1)
            dist.all_reduce(output, op=dist.ReduceOp.SUM)

        return output

