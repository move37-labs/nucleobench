"""
Integration tests for Saluki model.

To run:
```zsh
pytest -s integration_tests/saluki_integration_test.py
```
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from nucleobench.models.rna.saluki.model.saluki import Saluki
from nucleobench.models.rna.saluki.model_def import SalukiModel

TEST_DATA_PATH = (
    Path(__file__).parent.parent
    / "nucleobench"
    / "models"
    / "rna"
    / "saluki"
    / "testdata"
    / "saluki_test_samples.h5"
)


@pytest.fixture
def test_samples():
    """Fixture to provide the validation samples."""
    if not TEST_DATA_PATH.exists():
        pytest.skip(f"Test data is missing at {TEST_DATA_PATH}")

    with h5py.File(TEST_DATA_PATH, "r") as f:
        return {"test_in": f["test_in"][:], "keras_preds": f["keras_preds"][:]}


@pytest.mark.saluki
def test_saluki_accuracy_vs_golden_preds(test_samples):
    """Test the model accuracy on the extracted samples using the predict() method."""
    model = Saluki()

    x_batch = test_samples["test_in"]
    # Transpose if in old format (Batch, Channels, Length) -> (Batch, Length, Channels)
    if x_batch.shape[1] == 6 and x_batch.shape[2] == 12288:
        x_batch = np.transpose(x_batch, (0, 2, 1))

    expected_preds = test_samples["keras_preds"]

    # Decode the one-hot encoded tensor back into strings/metadata for predict()
    sequences = []
    cds_starts = []
    cds_ends = []
    exon_ends_list = []

    for i in range(x_batch.shape[0]):
        seq, start, end, exons = Saluki.decode_one_hot(x_batch[i])
        sequences.append(seq)
        cds_starts.append(start)
        cds_ends.append(end)
        exon_ends_list.append(exons)

    # Now test the ACTUAL predict method end-to-end
    actual_preds = model.predict(sequences, cds_starts, cds_ends, exon_ends_list)

    # Correlation Check
    corr = np.corrcoef(actual_preds.flatten(), expected_preds.flatten())[0, 1]
    print(f"Pearson Correlation: {corr:.4f}")

    # Check correlation threshold
    assert corr > 0.998, f"Pearson Correlation {corr:.4f} is below the threshold 0.998!"


@pytest.mark.saluki
def test_saluki_predict_api():
    """Test the predict method with real model to verify API handling."""
    saluki = Saluki()

    sequences = ["ACGT" * (12288 // 4)] * 2
    cds_starts = [0, 10]
    cds_ends = [1000, 1010]
    exon_ends = [[100, 200, 500], [150, 300, 600]]

    preds = saluki.predict(sequences, cds_starts, cds_ends, exon_ends)

    assert preds.shape == (2, 1)
    assert not np.isnan(preds).any()


@pytest.mark.saluki
def test_saluki_encoding():
    """Test the 6-track encoding logic."""
    saluki = Saluki()
    sequences = ["ACGT"]  # Length 4
    cds_starts = [0]
    cds_ends = [3]
    exon_ends = [[1, 3]]

    encoded = saluki._encode_batch(sequences, cds_starts, cds_ends, exon_ends)

    # Check shape (1, 12288, 6)
    assert encoded.shape == (1, 12288, 6)

    # Track 0-3: ACGT (indices 0, 1, 2, 3)
    # A -> [1,0,0,0] at pos 0
    assert encoded[0, 0, 0] == 1.0
    # C -> [0,1,0,0] at pos 1
    assert encoded[0, 1, 1] == 1.0
    # G -> [0,0,1,0] at pos 2
    assert encoded[0, 2, 2] == 1.0
    # T -> [0,0,0,1] at pos 3
    assert encoded[0, 3, 3] == 1.0

    # Track 4: CDS frame (start=0) index 4
    # Pos 0, 3, 6... should be 1
    assert encoded[0, 0, 4] == 1.0
    assert encoded[0, 3, 4] == 1.0

    # Track 5: Splice sites (ends=[1, 3]) index 5
    assert encoded[0, 1, 5] == 1.0
    assert encoded[0, 3, 5] == 1.0


@pytest.mark.saluki
def test_saluki_predict_5utr_api():
    """Test the predict_5utr convenience method."""
    saluki = Saluki()
    utrs = ["AA", "CC"]  # Two simple UTRs

    preds = saluki.predict_5utr(utrs)

    assert preds.shape == (2, 1)
    assert not np.isnan(preds).any()
    assert np.all((preds > -5.0) & (preds < 5.0))


@pytest.mark.saluki
def test_saluki_model_def_wrapper():
    """Test the SalukiModel wrapper class."""
    model = SalukiModel()
    utrs = ["AA", "CC"]

    preds = model(utrs)

    assert preds.shape == (2,)
    assert not np.isnan(preds).any()
    # Since SalukiModel returns -1 * predictions, let's verify
    # that the signs are inverted compared to direct predict_5utr.
    direct_preds = model.model.predict_5utr(utrs).flatten()
    assert np.allclose(preds, -1 * direct_preds)
