"""Tests for model_def.py

To test:
```zsh
pytest nucleobench/models/grelu/enformer/model_def_test.py
```
"""

import random
from unittest.mock import patch

import pytest
import torch

from nucleobench.common import testing_utils
from nucleobench.models.grelu.enformer import model_def

model_args = {
    "add_unsqueeze_to_output": True,
    "call_is_on_strings": False,
    "flip_sign": False,
    "extra_channels": 5313 - 1,
    "train_seq_len": 196_608,
}


@pytest.mark.parametrize("aggregation_type", ["muscle_not_liver"])
def test_model_def_sanity(aggregation_type):
    enformer_args = model_def.Enformer.debug_init_args()
    enformer_args["aggregation_type"] = aggregation_type
    m = model_def.Enformer(
        override_model=testing_utils.CountLetterModel(**model_args), **enformer_args
    )
    ret = m.inference_on_strings(["A" * 196608, "C" * 196608, "T" * 196608])
    assert list(ret.shape) == [3]


def _aggregation(model_out: torch.Tensor) -> torch.Tensor:
    assert model_out.ndim == 3
    return model_out[:, 0, 0]


def test_tism_correctness():
    """Check that TISM on an C-count network knows that Cs are important."""
    m = model_def.Enformer(
        override_model=testing_utils.CountLetterModel(**model_args),
        override_aggregation=_aggregation,
        **model_def.Enformer.debug_init_args(),
    )
    random.seed(0)
    base_str = [random.choice(["A", "C", "T", "G"]) for _ in range(196_608)]
    _, tism = m.tism(base_str)
    for base_nt, tism_dict in zip(base_str, tism):
        assert base_nt not in tism_dict
        if base_nt == "C":
            # Everything should be the same.
            assert tism_dict["A"] == tism_dict["T"] == tism_dict["G"]
            assert tism_dict["A"] > 0  # decrease the count, increase the energy.
        else:
            # TISM should show that the greatest change comes from adding a 'C'.
            for nt in ["A", "T", "G"]:
                if nt == base_nt:
                    continue
                assert tism_dict[nt] == 0  # changing to a non-C should be no change.
            assert tism_dict["C"] < 0


def test_tism_consistency():
    """TISM on a single nucleotide should be the same as the string.."""
    m = model_def.Enformer(
        override_model=testing_utils.CountLetterModel(**model_args),
        **model_def.Enformer.debug_init_args(),
    )
    random.seed(0)
    base_str = [random.choice(["A", "C", "T", "G"]) for _ in range(196_608)]

    v1, tism1 = m.tism(base_str)
    single_bp_tisms = [m.tism(base_str, [idx]) for idx in range(10)]

    for idx in range(len(single_bp_tisms)):
        v2, tism2 = single_bp_tisms[idx]
        assert v1 == v2
        assert len(tism2) == 1
        for k, v in tism2[0].items():
            assert v == tism1[idx][k]


@patch("torch.cuda.is_available")
@pytest.mark.parametrize("cuda_is_available", [True, False])
def test_device_placement(fake_torch_cuda, cuda_is_available):
    # Set mock for `torch.cuda.is_available`.
    fake_torch_cuda.return_value = cuda_is_available

    enformer_args = model_def.Enformer.debug_init_args()
    enformer_args["aggregation_type"] = "muscle_not_liver"
    m = model_def.Enformer(
        override_model=testing_utils.CountLetterModel(**model_args), **enformer_args
    )
    assert m.device == ("cuda" if cuda_is_available else "cpu")
    assert m.has_cuda is cuda_is_available


def test_k562_dnase_aggregation_shape():
    """k562_dnase returns one score per sequence in the batch."""
    m = model_def.Enformer(
        override_model=testing_utils.CountLetterModel(**model_args),
        aggregation_type="k562_dnase",
        track_indices=[0, 1],
        run_sanity_checks=False,
    )
    ret = m.inference_on_strings(["A" * 196608, "C" * 196608, "T" * 196608])
    assert list(ret.shape) == [3]


def test_k562_dnase_aggregation_math():
    """k562_dnase sums exactly the selected tracks and bins.

    We build the model, then call its aggregation closure directly on a
    synthetic tensor whose values we know exactly, so the test is independent
    of CountLetterModel internals and sequence length.
    """
    from nucleobench.models.grelu.enformer import constants

    n_tracks = len(constants.ENFORMER_TASKS_)
    batch, n_bins = 2, 8

    m_all_bins = model_def.Enformer(
        override_model=testing_utils.CountLetterModel(**model_args),
        aggregation_type="k562_dnase",
        track_indices=[0, 2],
        run_sanity_checks=False,
    )
    m_two_bins = model_def.Enformer(
        override_model=testing_utils.CountLetterModel(**model_args),
        aggregation_type="k562_dnase",
        track_indices=[0, 2],
        spatial_bins_to_aggregate=[1, 3],
        run_sanity_checks=False,
    )

    # All ones: sum over tracks [0,2] × all 8 bins = 2 * 8 = 16.0 per item.
    x = torch.ones(batch, n_tracks, n_bins)
    result_all = m_all_bins.aggregation(x)
    assert result_all.shape == (batch,)
    assert float(result_all[0]) == pytest.approx(16.0)

    # Same tensor, 2 selected bins: sum over tracks [0,2] × bins [1,3] = 2 * 2 = 4.0.
    result_two = m_two_bins.aggregation(x)
    assert result_two.shape == (batch,)
    assert float(result_two[0]) == pytest.approx(4.0)


def test_k562_dnase_requires_track_indices():
    """Omitting track_indices with aggregation_type='k562_dnase' raises ValueError."""
    with pytest.raises(ValueError, match="track_indices"):
        model_def.Enformer(
            override_model=testing_utils.CountLetterModel(**model_args),
            aggregation_type="k562_dnase",
            run_sanity_checks=False,
        )
