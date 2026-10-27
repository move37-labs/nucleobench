"""
To run:
```zsh
pytest -v -m borzoi_mouse integration_tests/borzoi_mouse_integration_test.py
```

Downloads the pinned mouse replicate 0 checkpoint (~700 MB) and mm10.
"""

import time

import numpy as np
import pytest
import torch

from nucleobench.models.grelu.borzoi.mouse.model_def import (
    OUTPUT_ORIGIN_,
    BorzoiMouse,
)

from .data_loaders import MouseEsExpressionByBucket


@pytest.fixture(scope="module")
def borzoi_mouse_predictions():
    print("\nLoading mouse Borzoi model...")
    # Exons are replaced per gene below; this is just a valid placeholder.
    model = BorzoiMouse(
        readout="es_e14",
        strand="+",
        exons=[(OUTPUT_ORIGIN_, OUTPUT_ORIGIN_ + 32)],
        run_sanity_checks=False,
    )
    print("Model loaded.")

    print("Loading sequences from data loader...")
    data_df = MouseEsExpressionByBucket().get_data()
    data_df = data_df[data_df["sequence_length"] == model.sequence_length]
    print("Sequences loaded.")

    predictions = {}
    for _, row in data_df.iterrows():
        # All four windows are on the + strand, so the gene body is a valid
        # single-exon interval in input coordinates.
        model.exons = [(int(row["gene_start"]), int(row["gene_end"]))]
        start_time = time.time()
        # The model is set to MINIMIZE, so we flip the sign back.
        predictions[row["gene"]] = -model([row["sequence"]]).item()
        print(
            f"  {row['gene']} prediction: {predictions[row['gene']]:.4f} "
            f"({time.time() - start_time:.2f} seconds)."
        )

    return {
        "model": model,
        "predictions": predictions,
        "pou5f1_sequence": data_df[data_df["gene"] == "POU5F1"]["sequence"].iloc[0],
        "pou5f1_exons": [
            (
                int(data_df[data_df["gene"] == "POU5F1"]["gene_start"].iloc[0]),
                int(data_df[data_df["gene"] == "POU5F1"]["gene_end"].iloc[0]),
            )
        ],
    }


@pytest.mark.borzoi_mouse
def test_es_genes_rank_above_liver_and_desert(borzoi_mouse_predictions):
    predictions = borzoi_mouse_predictions["predictions"]
    for es_gene in ["POU5F1", "NANOG"]:
        assert predictions[es_gene] > predictions["ALB"], (
            f"{es_gene} should be more expressed than ALB in ES cells"
        )
        assert predictions[es_gene] > predictions["GENE_DESERT"], (
            f"{es_gene} should be more expressed than a gene desert"
        )


@pytest.mark.borzoi_mouse
def test_es_genes_strongly_positive(borzoi_mouse_predictions):
    # Note: These thresholds may need to be tuned.
    predictions = borzoi_mouse_predictions["predictions"]
    assert predictions["POU5F1"] > 0.3, "Pou5f1 should be strongly expressed in ES"
    assert predictions["NANOG"] > 0.3, "Nanog should be strongly expressed in ES"


@pytest.mark.borzoi_mouse
def test_silent_loci_near_zero(borzoi_mouse_predictions):
    # Note: These thresholds may need to be tuned.
    predictions = borzoi_mouse_predictions["predictions"]
    assert predictions["ALB"] < 0.1, "ALB should be near-silent in ES cells"
    assert predictions["GENE_DESERT"] < 0.1, "A gene desert should be near-silent"


@pytest.mark.borzoi_mouse
def test_stochasticity(borzoi_mouse_predictions):
    model = borzoi_mouse_predictions["model"]
    model.exons = borzoi_mouse_predictions["pou5f1_exons"]
    sequence = borzoi_mouse_predictions["pou5f1_sequence"]
    pred1 = model([sequence]).item()
    pred2 = model([sequence]).item()
    assert abs(pred1 - pred2) < 1e-7, "Model predictions should be deterministic"


@pytest.mark.borzoi_mouse
def test_input_gradient_is_finite_and_nonzero(borzoi_mouse_predictions):
    model = borzoi_mouse_predictions["model"]
    model.exons = borzoi_mouse_predictions["pou5f1_exons"]
    x = model.string_to_onehot([borzoi_mouse_predictions["pou5f1_sequence"]])
    x.requires_grad_(True)
    model.inference_on_tensor(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


@pytest.mark.borzoi_mouse
def test_exon_score_matches_the_raw_profile(borzoi_mouse_predictions):
    """The scalar objective is the documented function of the raw profile."""
    from nucleobench.common.genomic_readout import coverage_mean

    model = borzoi_mouse_predictions["model"]
    exons = borzoi_mouse_predictions["pou5f1_exons"]
    model.exons = exons
    sequence = borzoi_mouse_predictions["pou5f1_sequence"]
    with torch.no_grad():
        profile = torch.as_tensor(
            model.predict_raw_on_strings([sequence])[:, model.track_index, :]
        )
        expected = -torch.log1p(
            coverage_mean(profile, exons, origin=OUTPUT_ORIGIN_, resolution=32)
        )
        actual = model.inference_on_tensor(model.string_to_onehot([sequence]))
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5)
