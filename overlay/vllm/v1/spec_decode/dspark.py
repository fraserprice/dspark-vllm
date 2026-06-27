# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import torch
from typing_extensions import override

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import VllmConfig
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.dspark_markov import sequential_markov_sample


class DSparkProposer(DFlashProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dspark"
        super().__init__(vllm_config, device, runner)
        self._dspark_anchor_token_ids: torch.Tensor | None = None

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        self._dspark_anchor_token_ids = next_token_ids.to(torch.long)
        num_query_total, sample_indices, new_cad = super().set_inputs_first_pass(
            target_token_ids,
            next_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            cad,
            num_rejected_tokens_gpu,
        )
        sample_indices.sub_(1)
        return num_query_total, sample_indices, new_cad

    @override
    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_logits = self._model_compute_logits(hidden_states, spec_step_idx)
        k = self.num_speculative_tokens
        v = base_logits.shape[-1]
        b = base_logits.shape[0] // k
        base_logits = base_logits.view(b, k, v)
        w1, w2 = self._markov_weights()
        assert self._dspark_anchor_token_ids is not None
        sample_fn = partial(
            self._sample_from_logits, sampling_metadata=sampling_metadata
        )
        tokens, probs = sequential_markov_sample(
            base_logits, self._dspark_anchor_token_ids, w1, w2, sample_fn
        )
        return tokens.reshape(b * k), (
            None if probs is None else probs.reshape(b * k, v)
        )

    @override
    def model_returns_tuple(self) -> bool:
        return False

    def _markov_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.model
        if isinstance(model, BreakableCUDAGraphWrapper):
            model = model.unwrap()
        return model.markov_w1, model.markov_w2
