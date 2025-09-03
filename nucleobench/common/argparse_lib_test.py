"""Tests for argparse_lib.py.

To test:
```zsh
pytest nucleobench/common/argparse_lib_test.py
```
"""
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd
import numpy as np

from nucleobench.common import argparse_lib

class ArgparseLibTest(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    # Tests for possibly_parse_start_sequence
    # ============================================

    def test_parse_start_sequence_no_prefix(self):
        """Test that a standard sequence string is returned unchanged."""
        seq = "ATGC"
        self.assertEqual(argparse_lib.possibly_parse_start_sequence(seq), "ATGC")

    def test_parse_start_sequence_from_local_file(self):
        """Test reading a sequence from a local file."""
        seq = "A" * 100
        tmp_file = os.path.join(self.test_dir.name, 'seq.txt')
        with open(tmp_file, 'w') as f:
            f.write(seq)
        
        path = f'local://{tmp_file}'
        self.assertEqual(argparse_lib.possibly_parse_start_sequence(path), seq)

    @mock.patch('nucleobench.common.argparse_lib.fetch_gcp_enformer_start_sequence_df')
    def test_parse_start_sequence_from_gcp(self, mock_fetch_df):
        """Test fetching a sequence from the mocked GCP DataFrame."""
        mock_df = pd.DataFrame({
            'sequence': ['ATGC', 'GATTACA'],
        }, index=[0, 1])
        mock_fetch_df.return_value = mock_df

        path = 'gcp_enformer://1'
        self.assertEqual(argparse_lib.possibly_parse_start_sequence(path), 'GATTACA')
        mock_fetch_df.assert_called_once()

    # Tests for possibly_parse_positions_to_mutate
    # ================================================

    def test_parse_positions_to_mutate_no_prefix(self):
        """Test parsing a comma-separated string of positions."""
        positions_str = "1,5,10"
        expected = [1, 5, 10]
        self.assertEqual(
            argparse_lib.possibly_parse_positions_to_mutate(positions_str), 
            expected
        )

    def test_parse_positions_to_mutate_from_local_file(self):
        """Test reading positions from a local file."""
        positions = [1, 10, 100]
        tmp_file = os.path.join(self.test_dir.name, 'pos.txt')
        with open(tmp_file, 'w') as f:
            f.write('\n'.join(map(str, positions)))
            
        path = f'local://{tmp_file}'
        self.assertEqual(argparse_lib.possibly_parse_positions_to_mutate(path), positions)

    @mock.patch('nucleobench.common.argparse_lib.fetch_gcp_enformer_start_sequence_df')
    def test_parse_positions_to_mutate_from_gcp(self, mock_fetch_df):
        """Test fetching positions from the mocked GCP DataFrame."""
        mock_df = pd.DataFrame({
            'positions_to_mutate': [np.array([1, 2]), np.array([3, 4])]
        }, index=[0, 1])
        mock_fetch_df.return_value = mock_df

        path = 'gcp_enformer://0'
        # The function should convert the numpy array to a list of ints
        self.assertEqual(argparse_lib.possibly_parse_positions_to_mutate(path), [1, 2])
        mock_fetch_df.assert_called_once()

    def test_parse_positions_to_mutate_empty_and_none(self):
        """Test that empty or None inputs return None."""
        self.assertIsNone(argparse_lib.possibly_parse_positions_to_mutate(None))
        self.assertIsNone(argparse_lib.possibly_parse_positions_to_mutate(''))
        self.assertIsNone(argparse_lib.possibly_parse_positions_to_mutate([]))
