"""Pytest unit tests for sequence_diversity_utils module.


To test:
```zsh
pytest nucleobench/common/sequence_diversity_utils_test.py
```
"""

import numpy as np
import pandas as pd
import pytest
from .sequence_diversity_utils import (
    pairwise_hamming_distance,
)


class TestPairwiseHammingDistance:
    """Test cases for pairwise_hamming_distance function."""

    def test_identical_sequences(self):
        """Two identical sequences should have distance 0.0."""
        test_seqs = ['ACGT', 'ACGT']
        result = pairwise_hamming_distance(test_seqs)
        assert result == 0.0

    def test_single_difference(self):
        """Two sequences differing by 1 position should have distance 1.0."""
        test_seqs = ['ACGT', 'ACGA']
        result = pairwise_hamming_distance(test_seqs)
        assert abs(result - 1.0) < 1e-6

    def test_multiple_differences(self):
        """Test sequences with multiple differences."""
        test_seqs = ['ACGT', 'TGCA']
        result = pairwise_hamming_distance(test_seqs)
        assert abs(result - 4.0) < 1e-6

    def test_three_sequences(self):
        """Test with three sequences."""
        test_seqs = ['ACGT', 'ACGA', 'ACGC']
        result = pairwise_hamming_distance(test_seqs)
        # Pairwise distances: (ACGT, ACGA)=1, (ACGT, ACGC)=1, (ACGA, ACGC)=1
        # Average = 1.0
        assert abs(result - 1.0) < 1e-6

    def test_empty_series(self):
        """Empty list should raise ValueError."""
        test_seqs = []
        with pytest.raises(ValueError, match='Must have at least 2 proposals'):
            pairwise_hamming_distance(test_seqs)

    def test_single_sequence(self):
        """Single sequence should raise ValueError (need at least 2 for pairwise)."""
        test_seqs = ['ACGT']
        with pytest.raises(ValueError, match='Must have at least 2 proposals'):
            pairwise_hamming_distance(test_seqs)