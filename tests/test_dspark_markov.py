import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "overlay"))

from vllm.v1.spec_decode.dspark_markov import (  # noqa: E402
    markov_bias,
    sequential_markov_sample,
)


def _greedy(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return logits.argmax(dim=-1), torch.softmax(logits, dim=-1)


def test_markov_bias_matches_explicit_low_rank_formula():
    vocab, rank = 5, 3
    w1 = torch.randn(vocab, rank, dtype=torch.float64)
    w2 = torch.randn(vocab, rank, dtype=torch.float64)
    prev = torch.tensor([2, 0, 4])
    got = markov_bias(prev, w1, w2)
    expected = torch.stack([w1[p] @ w2.t() for p in prev.tolist()])
    assert torch.allclose(got, expected)
    assert got.shape == (3, vocab)


def test_position_zero_is_conditioned_on_anchor_not_zero():
    vocab, rank = 4, 2
    base = torch.zeros(1, 1, vocab, dtype=torch.float64)
    w1 = torch.zeros(vocab, rank, dtype=torch.float64)
    w2 = torch.zeros(vocab, rank, dtype=torch.float64)
    w1[3] = torch.tensor([1.0, 0.0])
    w2[1] = torch.tensor([9.0, 0.0])
    anchor = torch.tensor([3])
    tokens, _ = sequential_markov_sample(base, anchor, w1, w2, _greedy)
    assert tokens.item() == 1


def test_sequential_dependency_chains_through_positions():
    vocab, rank = 4, 1
    base = torch.zeros(1, 3, vocab, dtype=torch.float64)
    w1 = torch.zeros(vocab, rank, dtype=torch.float64)
    w2 = torch.zeros(vocab, rank, dtype=torch.float64)
    w1[0, 0] = 1.0
    w1[1, 0] = 1.0
    w1[2, 0] = 1.0
    w2[1, 0] = 5.0
    w2[2, 0] = 0.0
    anchor = torch.tensor([0])
    tokens, _ = sequential_markov_sample(base, anchor, w1, w2, _greedy)
    assert tokens.squeeze(0).tolist() == [1, 1, 1]


def test_bias_actually_overrides_base_logits():
    vocab, rank = 3, 1
    base = torch.zeros(1, 1, vocab, dtype=torch.float64)
    base[0, 0, 0] = 10.0
    w1 = torch.zeros(vocab, rank, dtype=torch.float64)
    w2 = torch.zeros(vocab, rank, dtype=torch.float64)
    w1[7 % vocab, 0] = 1.0
    w2[2, 0] = 100.0
    anchor = torch.tensor([7 % vocab])
    tokens, _ = sequential_markov_sample(base, anchor, w1, w2, _greedy)
    assert tokens.item() == 2


def test_returned_probs_correspond_to_corrected_logits():
    vocab, rank = 6, 2
    base = torch.randn(2, 4, vocab, dtype=torch.float64)
    w1 = torch.randn(vocab, rank, dtype=torch.float64)
    w2 = torch.randn(vocab, rank, dtype=torch.float64)
    anchor = torch.tensor([1, 5])
    tokens, probs = sequential_markov_sample(base, anchor, w1, w2, _greedy)
    assert probs is not None and probs.shape == (2, 4, vocab)
    prev = anchor
    for pos in range(4):
        corrected = base[:, pos] + markov_bias(prev, w1, w2)
        assert torch.allclose(probs[:, pos], torch.softmax(corrected, dim=-1))
        prev = tokens[:, pos]


def test_probs_none_when_sample_fn_returns_none():
    vocab, rank = 4, 2
    base = torch.randn(1, 2, vocab, dtype=torch.float64)
    w1 = torch.randn(vocab, rank, dtype=torch.float64)
    w2 = torch.randn(vocab, rank, dtype=torch.float64)

    def argmax_only(logits: torch.Tensor) -> tuple[torch.Tensor, None]:
        return logits.argmax(dim=-1), None

    tokens, probs = sequential_markov_sample(
        base, torch.tensor([0]), w1, w2, argmax_only
    )
    assert probs is None and tokens.shape == (1, 2)


@pytest.mark.parametrize("batch,positions,vocab,rank", [(1, 1, 2, 1), (3, 5, 129280, 256)])
def test_shapes_hold_across_realistic_dims(batch, positions, vocab, rank):
    base = torch.randn(batch, positions, vocab)
    w1 = torch.randn(vocab, rank)
    w2 = torch.randn(vocab, rank)
    anchor = torch.randint(0, vocab, (batch,))
    tokens, probs = sequential_markov_sample(base, anchor, w1, w2, _greedy)
    assert tokens.shape == (batch, positions)
    assert probs is not None and probs.shape == (batch, positions, vocab)
