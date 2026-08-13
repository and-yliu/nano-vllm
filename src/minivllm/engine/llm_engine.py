import os

import torch
from transformers import AutoTokenizer, Qwen3Config

from minivllm.engine.model_runner import ModelRunner
from minivllm.engine.scheduler import Scheduler
from minivllm.engine.sequence import Sequence
from minivllm.sampling_params import SamplingParams


class LLMEngine:
    def __init__(
        self,
        model: str,
        block_size: int = 256,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 8192,
        max_model_length: int | None = None,
        gpu_memory_utilization: float = 0.9,
        device: torch.device | str = "cuda",
        dtype: torch.dtype | None = None,
    ):
        path = self._resolve(model)

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        config = Qwen3Config.from_pretrained(path)

        self.block_size = block_size
        self.runner = ModelRunner(
            config,
            path=path,
            block_size=block_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_length=max_model_length,
            device=device,
            dtype=dtype,
        )

        # The runner decides how many blocks fit only after it has weighed the
        # weights and a worst-case forward pass. Scheduler is created after ModelRunner
        self.scheduler = Scheduler(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_cached_blocks=self.runner.num_blocks,
            block_size=block_size,
            eos=self._stop_token_ids(path),
        )

        # reference Sequence through seq_id
        self.requests: dict[int, Sequence] = {}

        self.last_step_is_prefill = False
        self.last_step_num_tokens = 0

    def _stop_token_ids(self, path: str) -> set[int]:
        ids: set[int] = set()
        try:
            from transformers import GenerationConfig

            eos = GenerationConfig.from_pretrained(path).eos_token_id
            if isinstance(eos, int):
                ids.add(eos)
            elif eos:
                ids.update(eos)
        except Exception:
            pass

        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        return ids

    @staticmethod
    def _resolve(model: str) -> str:
        if os.path.isdir(model):
            return model
        from huggingface_hub import snapshot_download

        return snapshot_download(model)


    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
    ) -> int:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        seq = Sequence(
            token_ids=prompt,
            block_size=self.block_size,
            sampling_params=sampling_params or SamplingParams(),
        )
        self.requests[seq.seq_id] = seq
        self.scheduler.add(seq)
        return seq.seq_id


    def step(self) -> list[Sequence]:
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            self.last_step_num_tokens = 0
            return []

        self.last_step_is_prefill = is_prefill
        self.last_step_num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)

        token_ids = self.runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)

        return [seq for seq in seqs if seq.is_finished]

    def has_work(self) -> bool:
        return self.scheduler.has_work()


    def _output(self, seq: Sequence) -> dict:
        return {
            "seq_id": seq.seq_id,
            "token_ids": seq.output_token_ids,
            "text": self.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True),
        }

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        use_tqdm: bool = True,
    ) -> list[dict]:
        if isinstance(prompts, str):
            prompts = [prompts]

        seq_ids = [self.add_request(p, sampling_params) for p in prompts]

        bar = None
        if use_tqdm:
            from tqdm import tqdm

            bar = tqdm(total=len(seq_ids), desc="generating", unit="seq")

        outputs: dict[int, dict] = {}
        while self.has_work():
            for seq in self.step():
                outputs[seq.seq_id] = self._output(seq)
                del self.requests[seq.seq_id]
                if bar is not None:
                    bar.update(1)

        if bar is not None:
            bar.close()

        return [outputs[seq_id] for seq_id in seq_ids]
