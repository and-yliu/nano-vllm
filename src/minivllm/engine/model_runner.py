import torch
from transformers import Qwen3Config

from minivllm.engine.sequence import Sequence
from minivllm.layers.attention import Attention
from minivllm.model.qwen3 import Qwen3ForCausalLM
from minivllm.engine.sampler import sample
from minivllm.utils.context import set_context, reset_context
from minivllm.utils.loader import load_model


class ModelRunner:
    def __init__(
        self,
        config: Qwen3Config,
        path: str | None = None,
        block_size: int = 256,
        gpu_memory_utilization: float = 0.9,
        max_num_batched_tokens: int = 8192,
        max_model_length: int | None = None,
        device: torch.device | str = "cuda",
        dtype: torch.dtype | None = None,
    ):
        self.device = torch.device(device)

        # bf16 needs Ampere (sm_80+); on older cards it is emulated, slower, and
        # less accurate than fp16. Same rule as test_parity.
        if dtype is None:
            major, _ = torch.cuda.get_device_capability(self.device)
            dtype = torch.bfloat16 if major >= 8 else torch.float16
        self.dtype = dtype

        config.block_size = block_size
        self.config = config
        self.block_size = block_size

        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_length = min(
            max_model_length or config.max_position_embeddings,
            config.max_position_embeddings,
        )

        self.model = Qwen3ForCausalLM(config).to(device=self.device, dtype=self.dtype)
        if path is not None:
            load_model(self.model, path)
        self.model.eval()

        # Order matters. Warmup runs while the caches are still empty tensors,
        # so Attention skips its cache write and needs no block tables -- and it
        # is what tells allocate_kv_cache how much transient memory to reserve.
        self.warmup_model()
        self.num_blocks = self.allocate_kv_cache(gpu_memory_utilization)


    @torch.inference_mode()
    def warmup_model(self) -> None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        num_seqs = max(1, self.max_num_batched_tokens // self.max_model_length)
        seqs = [
            Sequence(token_ids=[0] * self.max_model_length, block_size=self.block_size)
            for _ in range(num_seqs)
        ]
        self.run(seqs, is_prefill=True)

        torch.cuda.empty_cache()

    def allocate_kv_cache(self, gpu_memory_utilization: float) -> int:
        free, total = torch.cuda.mem_get_info(self.device)
        stats = torch.cuda.memory_stats(self.device)
        peak = stats["allocated_bytes.all.peak"]
        current = stats["allocated_bytes.all.current"]

        available = total * gpu_memory_utilization - (total - free) - (peak - current)

        num_layers = self.config.num_hidden_layers
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim or (
            self.config.hidden_size // self.config.num_attention_heads
        )

        # 2 = K and V
        block_bytes = (
            2 * num_layers * self.block_size * num_kv_heads * head_dim * self.dtype.itemsize
        )
        num_blocks = int(available) // block_bytes
        assert num_blocks >= 1, (
            f"not enough free memory for even one KV block: {int(available)} bytes "
            f"available, {block_bytes} per block"
        )

        # zeros, not empty: a masked-off read of an untouched block should see 0,
        # not whatever NaN happened to be resident.
        kv_cache = torch.zeros(
            2, num_layers, num_blocks, self.block_size, num_kv_heads, head_dim,
            dtype=self.dtype, device=self.device,
        )

        # modules() yields in registration order, i.e. layer order
        layers = [m for m in self.model.modules() if isinstance(m, Attention)]
        assert len(layers) == num_layers, (
            f"found {len(layers)} Attention modules but config says {num_layers} layers"
        )
        for i, layer in enumerate(layers):
            layer.k_cache = kv_cache[0, i]
            layer.v_cache = kv_cache[1, i]

        self.kv_cache = kv_cache
        return num_blocks


    def _to_gpu(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(data, dtype=dtype, pin_memory=True).to(
            self.device, non_blocking=True
        )

    def _pad_block_tables(self, seqs: list[Sequence]) -> list[list[int]]:
        max_num_block = max(len(seq.block_table) for seq in seqs)
        return [
            seq.block_table + [-1] * (max_num_block - len(seq.block_table))
            for seq in seqs
        ]

    def _slots(self, seq: Sequence, start: int, end: int) -> list[int]:
        return [
            seq.block_table[i // self.block_size] * self.block_size + i % self.block_size
            for i in range(start, end)
        ]


    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        # the cumulative sequence length of each of the input sequences, length: num_seqs + 1
        cu_seqlens_q = [0]
        # the cumulative sequence length of each of the input sequences, length: num_seqs + 1
        cu_seqlens_k = [0]
        # the sequence length of each of the input sequences, length: num_seqs
        seqlens_q = []
        # the sequence length of each of the input sequences, length: num_seqs
        seqlens_k = []
        # what tokens are runned, length: sum of all input_ids after prefix cache
        input_ids = []
        # the physical position the tokens are in cache, length: same as input_ids
        slot_mappings = []
        # block_tables: num_seqs x num_blocks (padded)
        block_tables = []

        for seq in seqs:
            num_cached_tokens = seq.num_cached_tokens
            input_ids.extend(seq[num_cached_tokens:])
            seqlens_q.append(seq.num_tokens - num_cached_tokens)
            seqlens_k.append(seq.num_tokens)
            cu_seqlens_q.append(seqlens_q[-1] + cu_seqlens_q[-1])
            cu_seqlens_k.append(seqlens_k[-1] + cu_seqlens_k[-1])
            if seq.block_table:
                slot_mappings.extend(self._slots(seq, num_cached_tokens, seq.num_tokens))

        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            block_tables = self._pad_block_tables(seqs)

        set_context(
            is_prefill=True,
            cu_seqlens_q=self._to_gpu(cu_seqlens_q, torch.int32),
            cu_seqlens_k=self._to_gpu(cu_seqlens_k, torch.int32),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=self._to_gpu(slot_mappings, torch.long) if slot_mappings else None,
            context_lens=None,
            block_tables=self._to_gpu(block_tables, torch.int32) if block_tables else None,
        )

        return self._to_gpu(input_ids, torch.long)


    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids = []
        context_lens = []
        slot_mappings = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            context_lens.append(seq.num_tokens)
            slot_mappings.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )

        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=self._to_gpu(slot_mappings, torch.long),
            context_lens=self._to_gpu(context_lens, torch.long),
            block_tables=self._to_gpu(self._pad_block_tables(seqs), torch.int32),
        )

        return self._to_gpu(input_ids, torch.long)


    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, seqs: list[Sequence], is_prefill: bool
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids)

        if is_prefill:
            # Only the last token of each sequence predicts anything. Slice
            # before the LM head rather than after: logits for every token would
            # be [total_tokens, vocab] with vocab ~151k, almost all of it thrown
            # away.
            seqlens_q = [seq.num_tokens - seq.num_cached_tokens for seq in seqs]
            last = torch.cumsum(self._to_gpu(seqlens_q, torch.long), dim=0) - 1
            hidden_states = hidden_states[last]

        return self.model.compute_logits(hidden_states)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """One engine step: prepare -> forward -> sample."""
        if is_prefill:
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)

        logits = self.run_model(input_ids, seqs, is_prefill)
        token_ids = sample(logits, seqs)


        reset_context()
        return token_ids
