# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch


def markov_bias(
    prev_token_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
) -> torch.Tensor:
    low_rank = markov_w1.index_select(0, prev_token_ids)
    return torch.matmul(low_rank, markov_w2.t())


def sequential_markov_sample(
    base_logits: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    sample_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch_size, num_positions, vocab_size = base_logits.shape
    tokens = base_logits.new_empty((batch_size, num_positions), dtype=torch.int64)
    probs_buffer: torch.Tensor | None = None
    prev = anchor_token_ids
    for position in range(num_positions):
        logits = base_logits[:, position] + markov_bias(prev, markov_w1, markov_w2)
        sampled, probs = sample_fn(logits)
        tokens[:, position] = sampled
        if probs is not None:
            if probs_buffer is None:
                probs_buffer = base_logits.new_empty(
                    (batch_size, num_positions, vocab_size)
                )
            probs_buffer[:, position] = probs
        prev = sampled
    return tokens, probs_buffer
