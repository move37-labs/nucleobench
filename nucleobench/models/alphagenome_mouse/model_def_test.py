"""Offline tests for the AlphaGenome mouse readout and its JAX/PyTorch bridge."""

import numpy as np
import pandas as pd
import pytest
import torch

from nucleobench.models.alphagenome_mouse.bridge import JaxSelectedProfile
from nucleobench.models.alphagenome_mouse.model_def import AlphaGenomeMouse


def _metadata(strands=("+", "-"), ontology="EFO:0005483", assay="polyA plus RNA-seq"):
    """Mirrors the upstream RNA_SEQ metadata schema.

    The assay column is `Assay title`, with a space and a capital -- not
    `assay`. A fake that gets this wrong will agree with an adapter that gets
    it wrong too, so the names here have to match the real checkpoint.
    """
    return pd.DataFrame(
        {
            "name": [f"{ontology} {assay}"] * len(strands),
            "strand": list(strands),
            "Assay title": [assay] * len(strands),
            "ontology_curie": [ontology] * len(strands),
        }
    )


def _both_readouts_metadata():
    return pd.concat(
        [
            _metadata(ontology="EFO:0007075", assay="total RNA-seq"),
            _metadata(ontology="EFO:0005483", assay="polyA plus RNA-seq"),
        ],
        ignore_index=True,
    )


def test_selects_a_unique_assay_and_strand():
    model = AlphaGenomeMouse(
        "es_bruce4",
        "-",
        [(0, 10)],
        override_model=object(),
        override_metadata=_both_readouts_metadata(),
    )
    assert model.track_index == 3
    assert model.provenance["track_name"] == "EFO:0005483 polyA plus RNA-seq"


def test_rejects_metadata_without_a_unique_match():
    # No row for Bruce4 on the minus strand.
    with pytest.raises(ValueError, match="0 matches"):
        AlphaGenomeMouse(
            "es_bruce4",
            "-",
            [(0, 10)],
            override_model=object(),
            override_metadata=_both_readouts_metadata().iloc[:3],
        )
    duplicated = pd.concat([_metadata()] * 2, ignore_index=True)
    with pytest.raises(ValueError, match="2 matches"):
        AlphaGenomeMouse(
            "es_bruce4",
            "-",
            [(0, 10)],
            override_model=object(),
            override_metadata=duplicated,
        )


def test_rejects_unknown_readout_or_strand():
    with pytest.raises(ValueError, match="E14 or Bruce4"):
        AlphaGenomeMouse("nonexistent", "+", [(0, 10)], override_model=object())
    with pytest.raises(ValueError, match="E14 or Bruce4"):
        AlphaGenomeMouse("es_e14", "?", [(0, 10)], override_model=object())


def _injected_profile_model(exons):
    # The readout runs through the JAX bridge even with an injected profile.
    pytest.importorskip("jax")
    model = AlphaGenomeMouse(
        "es_e14",
        "+",
        exons,
        override_model=object(),
        override_metadata=_metadata(
            strands=("+",), ontology="EFO:0007075", assay="total RNA-seq"
        ),
    )
    # Stand in for the JAX trunk: the profile is just the `C` channel.
    model._jax_profile = lambda x: x[:, 1, :]
    return model


def test_objective_unions_overlapping_exons():
    # [(10, 12), (11, 13)] unions to [10, 13): three positions, two of them set.
    model = _injected_profile_model([(10, 12), (11, 13)])
    x = torch.zeros((1, 4, model.sequence_length))
    x[0, 1, 10:12] = 1
    assert model.inference_on_tensor(x).item() == pytest.approx(-np.log1p(2 / 3))


def test_objective_gradient_and_tism():
    model = _injected_profile_model([(10, 12), (11, 13)])
    x = torch.zeros((1, 4, model.sequence_length))
    x[0, 1, 10:13] = 1
    x.requires_grad_(True)
    score = model.inference_on_tensor(x)
    assert score.item() == pytest.approx(-np.log(2))
    score.backward()
    assert x.grad[0, 1, 10].item() == pytest.approx(-1 / 6)
    pairs, values = model.get_tism("A" * model.sequence_length, [10])
    assert pairs == [(10, "C"), (10, "G"), (10, "T")]
    assert values[0] < 0


def test_jax_bridge_vjp_matches_finite_differences():
    jnp = pytest.importorskip("jax.numpy")

    def function(value):
        return jnp.square(value).sum(axis=1)

    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    output = JaxSelectedProfile.apply(x, function)
    output.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [[2.0, 4.0, 6.0]])

    direction = torch.tensor([[0.5, -0.25, 0.75]])
    epsilon = 1e-3
    plus = JaxSelectedProfile.apply(x.detach() + epsilon * direction, function)
    minus = JaxSelectedProfile.apply(x.detach() - epsilon * direction, function)
    finite_difference = ((plus - minus) / (2 * epsilon)).item()
    assert finite_difference == pytest.approx(
        (x.grad * direction).sum().item(), rel=1e-3
    )


def test_rejects_metadata_missing_expected_columns():
    """The upstream schema uses `Assay title`; a renamed column must not pass."""
    renamed = _metadata().rename(columns={"Assay title": "assay"})
    with pytest.raises(ValueError, match="missing columns"):
        AlphaGenomeMouse(
            "es_bruce4",
            "-",
            [(0, 10)],
            override_model=object(),
            override_metadata=renamed,
        )


def test_reassigning_exons_rescores_the_new_window():
    """The cached coverage weights must not go stale when exons change."""
    model = _injected_profile_model([(10, 12)])
    x = torch.zeros((1, 4, model.sequence_length))
    x[0, 1, 10:12] = 1
    first = model.inference_on_tensor(x).item()

    # A window with no signal in it must score differently.
    model.exons = [(1000, 1002)]
    second = model.inference_on_tensor(x).item()

    assert first == pytest.approx(-np.log1p(1.0))
    assert second == pytest.approx(0.0)
    assert model.exons == [(1000, 1002)]


def test_exons_setter_validates():
    model = _injected_profile_model([(10, 12)])
    with pytest.raises(ValueError, match="outside"):
        model.exons = [(0, model.sequence_length + 1)]
