"""
To run:
```zsh
pytest integration_tests/borzoi_integration_test.py
```
"""
import time

import pytest

from nucleobench.models.grelu.borzoi import model_def

from .data_loaders import MuscleGeneExpressionByBucket


@pytest.fixture(scope="module")
def borzoi_predictions():
    # We are focusing on the center bins (2047 and 2048) because they are
    # most relevant for the TSS-centered sequences of CKM (muscle-specific)
    # and ALB (liver-specific).
    print("\nLoading Borzoi model (center bins only)...")
    borzoi_model = model_def.Borzoi(aggregation_type='muscle_not_liver',
                                    spatial_bins_to_aggregate=[3071, 3072],
                                    run_sanity_checks=False)
    print("Model loaded.")

    print("Loading sequences from data loader...")
    loader = MuscleGeneExpressionByBucket()
    data_df = loader.get_data()

    # Filter for borzoi-style sequences (524_288 length)
    seq_len = 524_288
    borzoi_data = data_df[data_df['sequence_length'] == seq_len]

    # Extract sequences
    ckm_sequence = borzoi_data[borzoi_data['gene'] == 'CKM']['sequence'].iloc[0]
    alb_sequence = borzoi_data[borzoi_data['gene'] == 'ALB']['sequence'].iloc[0]
    gene_desert_sequence = borzoi_data[borzoi_data['gene'] == 'GENE_DESERT']['sequence'].iloc[0]
    print("Sequences loaded.")

    # The model is set to MINIMIZE, so we flip the signs back.
    print("Running predictions...")

    start_time = time.time()
    ckm_pred = -borzoi_model([ckm_sequence]).item()
    end_time = time.time()
    print(f"  CKM prediction: {ckm_pred:.4f} ({end_time - start_time:.2f} seconds).")

    start_time = time.time()
    alb_pred = -borzoi_model([alb_sequence]).item()
    end_time = time.time()
    print(f"  ALB prediction: {alb_pred:.4f} ({end_time - start_time:.2f} seconds).")

    start_time = time.time()
    gene_desert_pred = -borzoi_model([gene_desert_sequence]).item()
    end_time = time.time()
    print(f"  Gene desert prediction: {gene_desert_pred:.4f} ({end_time - start_time:.2f} seconds).")

    print("Predictions complete.")

    return {
        'model': borzoi_model,
        'ckm_sequence': ckm_sequence,
        'ckm_pred': ckm_pred,
        'alb_pred': alb_pred,
        'gene_desert_pred': gene_desert_pred,
    }


@pytest.mark.borzoi
def test_relative_expression(borzoi_predictions):
    ckm_pred = borzoi_predictions['ckm_pred']
    alb_pred = borzoi_predictions['alb_pred']
    gene_desert_pred = borzoi_predictions['gene_desert_pred']
    assert ckm_pred > gene_desert_pred, "CKM should have higher expression than gene desert"
    assert gene_desert_pred > alb_pred, "Gene desert should have higher expression than ALB"


@pytest.mark.borzoi
def test_ckm_positive(borzoi_predictions):
    # Note: This threshold may need to be tuned.
    assert borzoi_predictions['ckm_pred'] > 35, "CKM expression should be strongly positive"


@pytest.mark.borzoi
def test_desert_neutral(borzoi_predictions):
    # Note: This threshold may need to be tuned.
    assert abs(borzoi_predictions['gene_desert_pred']) < 60, "Gene desert expression should be near zero"


@pytest.mark.borzoi
def test_alb_negative(borzoi_predictions):
    # Note: This threshold may need to be tuned.
    assert borzoi_predictions['alb_pred'] < -140, "ALB expression should be strongly negative"


@pytest.mark.borzoi
def test_stochasticity(borzoi_predictions):
    borzoi_model = borzoi_predictions['model']
    ckm_sequence = borzoi_predictions['ckm_sequence']
    pred1 = borzoi_model([ckm_sequence]).item()
    pred2 = borzoi_model([ckm_sequence]).item()
    assert abs(pred1 - pred2) < 1e-7, "Model predictions should be deterministic"
