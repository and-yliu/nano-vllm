import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from vllm.utils.context import get_context
from vllm.utils.distributed import get_rank, get_world_size

# turning tokens to embeddings
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embedding: int, embedding_dim: int):
        super().__init__()
        self.tp_size = get_world_size()
        self.tp_rank = get_rank()

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
        actual_shard_size = max(0, actual_end - actual_start)

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
            output = output * mask.unsqueeze(-1)
            dist.all_reduce(output, op=dist.ReduceOp.SUM)

        return output

# the last step to turn model output into logits for vocab
class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embedding: int, embedding_dim: int):
        super().__init__(num_embedding, embedding_dim)

    # x: [batch_size, seq_len, hidden_size]
    # weight: [vocab_size_per_partition, hidden_size]
    def forward(self, x: torch.Tensor):
        context = get_context()
        # in prefill many prompt token may be packed together, lgoits are for the final token of each prompt
        if context.is_prefill:
            # context.cu_seqlens_q = [0, 3, 5, 9]
            # last_token = [2, 4, 8]
            last_token = context.cu_seqlens_q[1:] - 1
            x = x[last_token].contiguous()

        # logits: [batch_size, seq_len, vocab_size_per_partition]
        # F.linear automatically transpose the weight
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # gpu 0 will hold copy of all logics from other gpus
            all_logits = [torch.empty(logits.size(), device=logits.device) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # collect logits from other gpu to gpu 0
            dist.gather(logits, gather_list=all_logits, dst=0)
            if self.tp_rank == 0:
                # concatenate logits
                # logits: [batch_size, seq_len, vocab_size]
                logits = torch.cat(all_logits, dim=-1)
                # trim to origignal size, ... means keep everything form all dimesions before the final one
                logits = logits[..., :self.num_embedding]

        return logits
