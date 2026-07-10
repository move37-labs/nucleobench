"""Tests for model_def.py

To test:
pytest nucleobench/models/malinois/model_def_test.py
"""

import numpy as np
import pytest
import torch

from nucleobench.common import string_utils, testing_utils
from nucleobench.models.malinois import model_def

model_args = {"extra_channels": 2, "call_is_on_strings": False}


def test_energy_calc_from_output_tensor_sanity():
    for target_feature in [0, 1, 2]:
        _ = model_def.energy_calc_from_output_tensor(
            torch.Tensor([[0.2, 0.3, 1.0], [1.5, 2.0, -20]]),
            target_feature=target_feature,
            bending_factor=1.0,
        )


def test_energy_calc_from_output_tensor_correctness():
    energy = model_def.energy_calc_from_output_tensor(
        torch.Tensor([[0.2, 0.3, 1.0], [1.5, 7.0, -20]]),
        target_feature=0,
        bending_factor=0.0,
        a_min=0.5,
        a_max=5,
    )
    ret = energy.detach().numpy()
    assert np.allclose(ret, np.array([1.0 - 0.5, 5.0 - 1.5]))


def test_malinois_sanity():
    m = model_def.Malinois(
        model_artifact=None,
        target_feature=0,
        bending_factor=1.0,
        override_model=testing_utils.CountLetterModel(**model_args),
    )
    ret = m.inference_on_strings(["AAA", "CCC", "TTT", "GGG", "ACT"])
    assert list(ret.shape) == [5]


def test_add_flank():
    m = model_def.Malinois(
        model_artifact=None,
        target_feature=0,
        bending_factor=1.0,
        override_model=testing_utils.CountLetterModel(**model_args),
        flank_length=2,
    )
    dnas = ["AAA", "CCC", "TTT", "GGG", "ACT"]
    tensor = string_utils.dna2tensor_batch(dnas)

    ret = m.add_flanks_tensor(tensor)
    assert list(ret.shape) == [5, 4, 7]


def test_grad_sanity():
    """Smoke test, and that tism inference is the same as normal inference."""
    m = model_def.Malinois(
        model_artifact=None,
        target_feature=0,
        bending_factor=1.0,
        override_model=testing_utils.CountLetterModel(**model_args),
    )
    y1, grad = m.tism("ATAAG")
    y2 = m.inference_on_strings(["ATAAG"])
    assert y1 == y2
    assert isinstance(grad, list)
    assert len(grad) == 5


def test_tism_correctness():
    """Check that TISM on an C-count network knows that Cs are important."""
    m = model_def.Malinois(
        model_artifact=None,
        target_feature=0,
        bending_factor=0.0,
        a_min=None,
        a_max=None,
        vocab=["A", "C", "G", "T"],
        override_model=testing_utils.CountLetterModel(vocab_i=1, **model_args),
    )
    base_str = "ATCCA"
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
    m = model_def.Malinois(
        model_artifact=None,
        target_feature=0,
        bending_factor=0.0,
        a_min=None,
        a_max=None,
        vocab=["A", "C", "G", "T"],
        override_model=testing_utils.CountLetterModel(vocab_i=1, **model_args),
    )
    base_str = "ATCCA"
    v1, tism1 = m.tism(base_str)
    single_bp_tisms = [m.tism(base_str, [idx]) for idx in range(len(base_str))]

    for idx in range(len(single_bp_tisms)):
        v2, tism2 = single_bp_tisms[idx]
        assert v1 == v2
        assert len(tism2) == 1
        for k, v in tism2[0].items():
            assert v == tism1[idx][k]


@pytest.mark.parametrize("flank_length", [0, 100, 200])
def test_flank_length(flank_length: int):
    """Check that inference works with various flank sizes."""
    seq_len = 600 - 2 * flank_length

    m = model_def.Malinois(
        model_artifact=None,
        flank_length=flank_length,
        target_feature=0,
        bending_factor=0.0,
        a_min=None,
        a_max=None,
        vocab=["A", "C", "G", "T"],
        override_model=testing_utils.CountLetterModel(**model_args),  # type: ignore[arg-type]
        check_input_shape=True,
    )
    m(["A" * seq_len])


def test_tism_torch_correctness():
    model = model_def.Malinois(
        target_feature=0,
        bending_factor=0.0,
        vocab=["A", "C"],
        flank_length=0,
        override_model=testing_utils.CountLetterModel(
            vocab_i=1, vocab_len=2, **model_args
        ),
    )
    seq = "AAC"
    tism_tensor = model.tism_torch(seq)
    assert isinstance(tism_tensor, torch.Tensor)
    assert tism_tensor.shape == (2, 3)
    # Reference base is exactly zero
    base_seq_idx = [0, 0, 1]  # 'A', 'A', 'C' in vocab ['A', 'C']
    for i, ref in enumerate(base_seq_idx):
        assert tism_tensor[ref, i] == 0.0


def test_tism_torch_consistency_with_tism():
    vocab = ["A", "C"]
    model = model_def.Malinois(
        target_feature=0,
        bending_factor=0.0,
        flank_length=0,
        vocab=vocab,
        override_model=testing_utils.CountLetterModel(
            vocab_i=1, vocab_len=len(vocab), **model_args
        ),
    )
    seq = "AAC"
    _, sg_dicts = model.tism(seq)
    tism_tensor = model.tism_torch(seq)

    for i, sg_dict in enumerate(sg_dicts):
        for j, nt in enumerate(vocab):
            val_torch = float(tism_tensor[j, i])
            val_py = float(sg_dict.get(nt, 0.0))
            assert abs(val_torch - val_py) < 1e-6
