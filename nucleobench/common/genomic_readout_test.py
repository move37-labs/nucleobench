"""Tests for shared genomic input validation and exon coverage."""

import pytest
import torch

from nucleobench.common.genomic_readout import (
    coverage_mean,
    coverage_weights,
    one_hot,
    union_intervals,
    validate_tensor,
    weighted_mean,
)


def test_interval_union_and_partial_bins():
    assert union_intervals([(4, 10), (0, 6), (12, 14)], 0, 20) == [(0, 10), (12, 14)]
    profile = torch.tensor([[2.0, 6.0, 10.0]])
    # Bins are [0,4), [4,8), [8,12). The union [1,7) covers 3 bp of each of the
    # first two bins, so the mean is weighted 3:3 and ignores the third bin.
    result = coverage_mean(profile, [(1, 5), (4, 7)], origin=0, resolution=4)
    assert result.item() == pytest.approx((3 * 2 + 3 * 6) / 6)
    with pytest.raises(ValueError):
        coverage_mean(profile, [(0, 13)], origin=0, resolution=4)


def test_precomputed_weights_match_coverage_mean():
    profile = torch.tensor([[2.0, 6.0, 10.0], [1.0, 1.0, 1.0]])
    exons = [(1, 5), (4, 7)]
    weights = coverage_weights(profile.shape[1], exons, origin=0, resolution=4)
    assert torch.equal(weights, torch.tensor([3.0, 3.0, 0.0]))
    assert torch.allclose(
        weighted_mean(profile, weights),
        coverage_mean(profile, exons, origin=0, resolution=4),
    )
    # Weights built for a different grid are rejected rather than broadcast.
    with pytest.raises(ValueError):
        weighted_mean(profile, torch.ones(4))


def test_interval_union_rejects_empty_and_out_of_bounds():
    with pytest.raises(ValueError):
        union_intervals([], 0, 20)
    with pytest.raises(ValueError):
        union_intervals([(5, 5)], 0, 20)
    with pytest.raises(ValueError):
        union_intervals([(-1, 5)], 0, 20)


def test_one_hot_lowercase_and_n():
    value = one_hot(["acgn"], 4)
    assert value.shape == (1, 4, 4)
    assert torch.equal(value[0], one_hot(["ACGN"], 4)[0])
    assert value[0, :, 3].sum() == 0
    with pytest.raises(ValueError):
        one_hot(["ACRX"], 4)
    with pytest.raises(ValueError):
        one_hot(["ACG"], 4)
    with pytest.raises(ValueError):
        one_hot([], 4)


def test_validate_tensor():
    validate_tensor(torch.zeros((2, 4, 8)), 8)
    with pytest.raises(ValueError):
        validate_tensor(torch.zeros((2, 4, 7)), 8)
    with pytest.raises(ValueError):
        validate_tensor(torch.zeros((4, 8)), 8)
    with pytest.raises(ValueError):
        validate_tensor(torch.full((1, 4, 8), float("nan")), 8)
