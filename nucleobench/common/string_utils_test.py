"""Test utils.py

To test:
pytest nucleobench/common/string_utils_test.py
"""

import numpy as np
from nucleobench.common import string_utils as utils

def test_dna2tensor2dna():
    """Test roundtrip conversion between DNA and tensor."""
    for cur_dna in [
        'ACGT',
        'ACGTACGT',
        'ACGTACGTACGT',
        'ACGTACGTACGTACGT',
        'ACGTACGTACGTACGTACGT',
        'ACGTACGTACGTACGTACGTACGT',
        'ACGTACGTACGTACGTACGTACGTACGT',
        'ACGTACGTACGTACGTACGTACGTACGTACGT',
    ]:
        cur_tensor = utils.dna2tensor(cur_dna)
        roundtrip_dna = utils.tensor2dna(cur_tensor)
        assert cur_dna == roundtrip_dna


def test_tensor2dna_batch():
    """Test roundtrip conversion between tensor and DNA, with batch util."""
    test_dnas = [
        'ACGTACGTACGT',
        'CGTACGTACGTA',
        'GTACGTACGTAA',
        'TACGTACGTCCC',
    ]
    tensors = utils.dna2tensor_batch(test_dnas)
    roundtrip_dnas = utils.tensor2dna_batch(tensors)
    for cur_dna, cur_roundtrip_dna in zip(test_dnas, roundtrip_dnas):
        assert cur_dna == cur_roundtrip_dna


def test_load_sequences():
    """Test the string codepaths."""
    # TODO(joelshor): Add tests for the file reading.
    dna_str = 'ATGCTGG'
    dna_strs = ['ATGCCCG', 'ATAA', 'ATTGC']

    assert utils.load_sequences(dna_str)[0] == dna_str
    assert utils.load_sequences(','.join(dna_strs))[0] == dna_strs