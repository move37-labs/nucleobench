"""
Integration test for Enformer model.

To run:
```zsh
pytest -s integration_tests/enformer_integration_test.py
```
"""
import time

import pytest

from nucleobench.models.grelu.enformer import model_def
from .data_loaders import MuscleGeneExpressionByBucket


@pytest.fixture(scope="module")
def enformer_predictions():
    # We are focusing on the center bins (447 and 448) because they are
    # most relevant for the TSS-centered sequences of CKM (muscle-specific)
    # and ALB (liver-specific), which are loaded via MuscleGeneExpressionByBucket data loader.
    print("\nLoading Enformer model (center bins only)...")
    model = model_def.Enformer(
        aggregation_type='muscle_not_liver',
        spatial_bins_to_aggregate=[447, 448],
        run_sanity_checks=False,
    )
    print("Model loaded.")

    print("Loading sequences from data loader...")
    loader = MuscleGeneExpressionByBucket()
    data_df = loader.get_data()

    # Filter for enformer-style sequences (196_608 length)
    seq_len = 196_608
    enformer_data = data_df[data_df['sequence_length'] == seq_len]

    ckm_sequence = enformer_data[enformer_data['gene'] == 'CKM']['sequence'].iloc[0]
    alb_sequence = enformer_data[enformer_data['gene'] == 'ALB']['sequence'].iloc[0]
    gene_desert_sequence = enformer_data[enformer_data['gene'] == 'GENE_DESERT']['sequence'].iloc[0]
    print("Sequences loaded.")

    # The model is set to MINIMIZE, so we flip the signs back.
    print("Running predictions...")

    start_time = time.time()
    ckm_pred = -model([ckm_sequence]).item()
    print(f"  CKM prediction: {ckm_pred:.4f} ({time.time() - start_time:.2f} seconds).")

    start_time = time.time()
    alb_pred = -model([alb_sequence]).item()
    print(f"  ALB prediction: {alb_pred:.4f} ({time.time() - start_time:.2f} seconds).")

    start_time = time.time()
    gene_desert_pred = -model([gene_desert_sequence]).item()
    print(f"  Gene desert prediction: {gene_desert_pred:.4f} ({time.time() - start_time:.2f} seconds).")

    print("Predictions complete.")

    return {
        "model": model,
        "ckm_sequence": ckm_sequence,
        "ckm_pred": ckm_pred,
        "alb_pred": alb_pred,
        "gene_desert_pred": gene_desert_pred,
    }


@pytest.mark.enformer
def test_relative_expression(enformer_predictions):
    preds = enformer_predictions
    assert preds["ckm_pred"] > preds["gene_desert_pred"], \
        "CKM should have higher expression than gene desert"
    assert preds["gene_desert_pred"] > preds["alb_pred"], \
        "Gene desert should have higher expression than ALB"


@pytest.mark.enformer
def test_ckm_positive(enformer_predictions):
    assert enformer_predictions["ckm_pred"] > 3000, \
        "CKM expression should be strongly positive"


@pytest.mark.enformer
def test_desert_neutral(enformer_predictions):
    assert abs(enformer_predictions["gene_desert_pred"]) < 100, \
        "Gene desert expression should be near zero"


@pytest.mark.enformer
def test_alb_negative(enformer_predictions):
    assert enformer_predictions["alb_pred"] < -1000, \
        "ALB expression should be strongly negative"


@pytest.mark.enformer
def test_stochasticity(enformer_predictions):
    model = enformer_predictions["model"]
    ckm_sequence = enformer_predictions["ckm_sequence"]
    pred1 = model([ckm_sequence]).item()
    pred2 = model([ckm_sequence]).item()
    assert pred1 == pytest.approx(pred2, abs=1e-7), \
        "Model predictions should be deterministic"