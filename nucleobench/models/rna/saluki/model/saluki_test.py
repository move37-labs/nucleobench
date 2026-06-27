from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from nucleobench.models.rna.saluki.model.saluki import Saluki
from nucleobench.models.rna.saluki.model.reporter_constants import BGH_3UTR, EGFP_CDS


def test_saluki_encode_batch():
    with patch.object(Saluki, "_load_model", return_value=None):
        model = Saluki()
        
        # Define some test inputs
        sequences = ["ACGT", "AACCGGTT"]
        cds_starts = [1, 2]
        cds_ends = [3, 5]
        exon_ends = [[2], [4, 6]]
        
        encoded = model._encode_batch(sequences, cds_starts, cds_ends, exon_ends)
        
        # Check shape: (Batch, Length, Channels)
        assert encoded.shape == (2, 12288, 6)
        
        # Check nucleotide encoding for first sequence: "ACGT" -> channels 0, 1, 2, 3
        # A
        assert encoded[0, 0, 0] == 1.0
        # C
        assert encoded[0, 1, 1] == 1.0
        # G
        assert encoded[0, 2, 2] == 1.0
        # T
        assert encoded[0, 3, 3] == 1.0
        
        # Check CDS encoding (channel 4) for first sequence:
        # cds_start = 1, cds_end = 3. Every 3rd position from start to end.
        # start=1, start+3=4 (but end is 3, so only 1)
        assert encoded[0, 1, 4] == 1.0
        assert encoded[0, 2, 4] == 0.0
        
        # Check exon ends encoding (channel 5) for first sequence:
        # exon_ends = [2]
        assert encoded[0, 2, 5] == 1.0
        assert encoded[0, 3, 5] == 0.0


def test_saluki_decode_one_hot():
    # Create a dummy one-hot array of shape (12288, 6)
    x = np.zeros((12288, 6), dtype=np.float32)
    # Encode "ACGT" at the beginning
    x[0, 0] = 1.0  # A
    x[1, 1] = 1.0  # C
    x[2, 2] = 1.0  # G
    x[3, 3] = 1.0  # T
    
    # CDS start at 1, end at 3
    x[1, 4] = 1.0
    x[3, 4] = 1.0
    
    # Exon ends at 2
    x[2, 5] = 1.0
    
    seq, cds_start, cds_end, exon_ends = Saluki.decode_one_hot(x)
    
    # The rest of the sequence will be 'N'
    assert seq.startswith("ACGT")
    assert len(seq) == 12288
    assert seq[4:] == "N" * (12288 - 4)
    assert cds_start == 1
    assert cds_end == 3
    assert exon_ends == [2]


def test_saluki_decode_one_hot_no_cds():
    x = np.zeros((12288, 6), dtype=np.float32)
    seq, cds_start, cds_end, exon_ends = Saluki.decode_one_hot(x)
    assert cds_start == -1
    assert cds_end == -1
    assert exon_ends == []


def test_saluki_predict_5utr():
    with patch.object(Saluki, "_load_model", return_value=None):
        model = Saluki()
        model.predict = MagicMock(return_value=np.array([0.5, 0.8]))
        
        utrs = ["AAA", "CCC"]
        preds = model.predict_5utr(utrs)
        
        assert np.array_equal(preds, np.array([0.5, 0.8]))
        
        # Verify predict was called with correct constructed sequences and metadata
        model.predict.assert_called_once()
        args, kwargs = model.predict.call_args
        
        sequences, cds_starts, cds_ends, exon_ends_list = args
        
        assert len(sequences) == 2
        assert sequences[0] == "AAA" + EGFP_CDS + BGH_3UTR
        assert sequences[1] == "CCC" + EGFP_CDS + BGH_3UTR
        
        assert cds_starts == [3, 3]
        assert cds_ends == [3 + len(EGFP_CDS), 3 + len(EGFP_CDS)]
        assert exon_ends_list == [[], []]


def test_saluki_predict():
    with patch.object(Saluki, "_load_model", return_value=None):
        model = Saluki()
        model.model = MagicMock()
        model.model.predict = MagicMock(return_value=np.array([[0.5], [0.8]]))
        
        # Test predict with shift logic (leading Ns)
        sequences = ["NNNACGT", "AACCGGTT"]
        cds_starts = [4, 2]
        cds_ends = [6, 5]
        exon_ends = [[5], [4, 6]]
        
        preds = model.predict(sequences, cds_starts, cds_ends, exon_ends)
        assert np.array_equal(preds, np.array([[0.5], [0.8]]))
        model.model.predict.assert_called_once()
