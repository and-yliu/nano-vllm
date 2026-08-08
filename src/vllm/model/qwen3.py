import torch
import torch.nn as nn
from transformers import Qwen3Config

from vllm.layers.linear import QKVMergedColumnParallelLinear, RowParallelLinear, MergeColumnParallelLinear
from vllm.layers.attention import Attention
from vllm.layers.activation import SiluAndMul
from vllm.layers.layernorm import RMSNorm
from vllm.layers.rope import RotaryEmbedding
from vllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from vllm.utils.distributed import get_world_size


class Qwen3Attention(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        num_head: int,
        head_dim: int,
        num_kv_heads: int = None,
        qkv_bias: bool = None,
        base: int = 10000,
        rms_norm_eps: float = 1e-5,
        max_position: int = 32768,
        block_size: int = 256
    ):
        super().__init__()
        self.tp_size = get_world_size()

        # num_head is per GPU
        self.total_num_heads = num_head
        assert self.total_num_heads % self.tp_size == 0
        self.num_head = self.total_num_heads // self.tp_size

        # num_kv_head is per GPU
        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_head
        assert self.total_num_kv_heads % self.tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_head

        self.qkv_linear = QKVMergedColumnParallelLinear(
            input_size=hidden_size, 
            head_size=self.head_dim, 
            num_heads=self.total_num_heads, 
            num_kv_heads=self.total_num_kv_heads,
            bias = qkv_bias
        )

        self.q_size = self.head_dim * self.num_head
        self.kv_size = self.head_dim * self.num_kv_heads
        self.qkv_bias = qkv_bias

        self.row_linear = RowParallelLinear(
            input_size=self.head_dim * self.total_num_heads,
            output_size=hidden_size
        )

        self.rotary_embedding = RotaryEmbedding(base=base, embed_dim=self.head_dim, max_position=max_position)
        self.q_rms_norm = RMSNorm(torch.ones(head_dim), rms_norm_eps)
        self.k_rms_norm = RMSNorm(torch.ones(head_dim), rms_norm_eps)

        self.attention = Attention(
            head_dim=self.head_dim,
            num_q_head=self.num_head,
            num_kv_head=self.num_kv_heads,
            block_size=block_size
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        # Input: hidden_states, shape (B, N, hidden_size) in all gpu

        #output shape per GPU (B, N, self.head_dim * (self.num_head + 2 * self.num_kv_head))
        qkv = self.qkv_linear(hidden_states)
        # separte QKV with per GPU size
        # q size (B, N, self.head_dim * self.num_head )
        # k size (B, N, self.head_dim * self.num_kv_head )
        # v size (B, N, self.head_dim * self.num_kv_head )
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)

        # reshape the QKV to separate the last dimension
        if q.dim() == 2:
            q = q.view(-1, self.num_head, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
        else:
            batch, token = q.shape[0], q.shape[1]
            q = q.view(batch, token, self.num_head, self.head_dim)
            k = k.view(batch, token, self.num_kv_heads, self.head_dim)
            v = v.view(batch, token, self.num_kv_heads, self.head_dim)

        # apply rms_norm nor to Q and K which is used in attention weight multiplication
        # It helps to avoid big numbers that cause softmax instability
        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)

        # add position information to q and k
        q, k = self.rotary_embedding(positions, q, k)

        # output shape per gpu: (batch*token, num_head, head_dim)
        output = self.attention(q, k ,v)

        # gathering all (batch*token num_heads * head_dim) sharded across all GPU
        # output sahpe (batch*token, hidden_size = (self.total_num_heads * self.head_dim))
        return self.row_linear(output)


class Qwen3MLP(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.activation = SiluAndMul()
        self.gate_up = MergeColumnParallelLinear(
            input_size=hidden_size,
            output_size=[intermediate_size, intermediate_size],
            bias = bias
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias = bias
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        x = self.gate_up(x)
        x = self.activation(x)
        return self.down_proj(x)

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.attention = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_head=config.num_attention_heads,
            head_dim=config.head_dim,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            qkv_bias=config.attention_bias,
            rms_norm_eps=config.rms_norm_eps,
        )

        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=False
        )

        self.pre_norm = RMSNorm(
            gamma=torch.ones(config.hidden_size),
            eps=config.rms_norm_eps
        )

        self.post_attention_norm = RMSNorm(
            gamma=torch.ones(config.hidden_size),
            eps=config.rms_norm_eps
        )

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        # normalized and check if residual is given
        if residual is None:
            hidden_states, residual = self.pre_norm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.pre_norm(hidden_states, residual)

        from vllm.utils.context import get_context
        # get positions of hidden_states
        context = get_context()
        #  when prefill, get postion for all token in the sequence
        if context.is_prefill and context.cu_seqlens_q is not None:
            positions = []
            cu_seqlens = context.cu_seqlens_q.cpu().tolist()
            for i in range(len(cu_seqlens) - 1):
                seqlen = cu_seqlens[i+1] - cu_seqlens[i]
                positions += range(seqlen)
            positions = torch.tensor(positions, dtype=torch.long, device=hidden_states.device)
        elif context.is_prefill:
            # rotate all previous tokens
            positions = torch.arange(hidden_states.size(0), device=hidden_states.device)
        else: 
            # decoding just the last token
            positions = context.context_lens - 1

        hidden_states = self.attention(positions, hidden_states)
        hidden_states, residual = self.post_attention_norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embedding = VocabParallelEmbedding(
            num_embedding=config.vocab_size,
            embedding_dim=config.hidden_size
        )
        self.decoder_layers = nn.ModuleList(
            [Qwen3DecoderLayer(
                config=config
            ) for _ in range(config.num_hidden_layers)]
        )
        self.rms_norm = RMSNorm(
            gamma=torch.ones(config.hidden_size),
            eps=config.rms_norm_eps
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # turn tokens into embedding 
        x = self.embedding(token_ids)
        # run it through each decoder layer
        residual = None
        for layer in self.decoder_layers:
            x, residual = layer(x, residual)

        # nomalize the final output
        output, _ = self.rms_norm(x, residual)
        return output

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(
            num_embedding=config.vocab_size,
            embedding_dim=config.hidden_size
        )
        # using the same weight matrix for converting input token to embedding and convert hidden states back to vocab logits
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embedding.weight.data

    # run the model, by giving the token_ids
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model(token_ids)

    # from the output of the model, get the logits for each vocab
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)




        
