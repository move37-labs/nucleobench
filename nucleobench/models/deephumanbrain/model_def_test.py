"""Offline tests for DeepHumanBrain, using an injected model."""

import numpy as np
import pytest
import torch

from nucleobench.common.genomic_readout import one_hot
from nucleobench.models.deephumanbrain.model_def import N_OUTPUTS_, DeepHumanBrain

LABELS_ = [f"label_{i}" for i in range(N_OUTPUTS_)]


class TinyBrain(torch.nn.Module):
    """Scores every output by the `C` content of the (batch, length, 4) input."""

    def forward(self, x, training=False):
        assert training is False
        score = x[:, :, 1].sum(dim=1)
        return score[:, None].expand(-1, N_OUTPUTS_)


def _model(output_names=None, labels=None):
    return DeepHumanBrain(
        output_names if output_names is not None else [LABELS_[0]],
        override_model=TinyBrain(),
        override_output_names=LABELS_ if labels is None else labels,
        device="cpu",
    )


def test_string_and_tensor_paths_agree():
    model = _model()
    sequence = "a" * model.sequence_length
    assert model([sequence]).shape == (1,)
    x = one_hot([sequence], model.sequence_length)
    np.testing.assert_allclose(model([sequence]), model.inference_on_tensor(x).numpy())
    assert model.predict_raw_on_strings([sequence]).shape == (1, N_OUTPUTS_)


def test_input_gradient_sign():
    model = _model()
    x = one_hot(["a" * model.sequence_length], model.sequence_length)
    x.requires_grad_(True)
    model.inference_on_tensor(x).sum().backward()
    # More `C` means more accessibility, so a lower (better) score.
    assert x.grad[0, 1, 0] == -1
    assert model.provenance["selected_output_names"] == [LABELS_[0]]


def test_tism_matches_the_input_gradient():
    model = _model()
    sequence = "A" * model.sequence_length
    pairs, values = model.get_tism(sequence, [0])
    assert pairs == [(0, "C"), (0, "G"), (0, "T")]
    np.testing.assert_allclose(values, [-1, 0, 0])
    all_pairs, all_values = model.get_tism(sequence)
    assert all_pairs[:3] == pairs
    np.testing.assert_allclose(all_values[:3], values)


def test_rejects_unknown_or_duplicate_labels():
    with pytest.raises(ValueError, match="exact CREsted output labels"):
        _model(output_names=["unknown"])
    with pytest.raises(ValueError, match="unique labels"):
        _model(labels=["duplicate"] * N_OUTPUTS_)
    with pytest.raises(ValueError, match="unique labels"):
        _model(labels=LABELS_[:-1])


def test_injected_model_requires_explicit_labels():
    with pytest.raises(ValueError, match="exact output names"):
        DeepHumanBrain([LABELS_[0]], override_model=TinyBrain())


def test_strings_require_the_exact_input_length():
    with pytest.raises(ValueError, match="length"):
        _model()(["ACGT"])
