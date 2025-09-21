"""Tests for gcp_utils.py.

To test:
```zsh
pytest nucleobench/common/gcp_utils_test.py
```
"""
import argparse
import os
import pickle
import tempfile
import unittest
from unittest import mock
import io
import pandas as pd
import numpy as np
import collections

from nucleobench.common import gcp_utils

# Define the mock ParsedArgs structure at the module level so pickle can find it.
ParsedArgs = collections.namedtuple('ParsedArgs', ['main_args', 'model_init_args', 'opt_init_args'])

class GcpUtilsTest(unittest.TestCase):

    def test_save_proposals_with_parsed_args(self):
        """
        Tests that `save_proposals` can correctly flatten a complex, nested
        object like the ParsedArgs example.
        """
        # Create the complex nested object.
        main_args = argparse.Namespace(
            model='malinois',
            optimization='directed_evolution',
            max_number_of_rounds=1,
            start_sequence='ATATAT', # Shortened for brevity
        )
        parsed_args_obj = ParsedArgs(
            main_args=main_args,
            model_init_args=None,
            opt_init_args=None
        )

        # The function expects a list of dictionaries.
        write_dicts = [{'run_data': parsed_args_obj, 'exp_starttime_str': '20231102-110000'}]
        
        # Args for the save_proposals function itself.
        save_args = argparse.Namespace(optimization='test_opt', model='test_model')

        for format in ['pkl', 'parquet']:
            with self.subTest(format=format):
                output_dir = os.path.join(self.test_dir.name, format)
                gcp_utils.save_proposals(write_dicts, save_args, output_dir, format=format)
                
                # Find and read the output file
                expected_subdir = os.path.join(output_dir, 'test_opt_test_model', '20231102-110000')
                self.assertTrue(os.path.isdir(expected_subdir))
                file_path = os.path.join(expected_subdir, os.listdir(expected_subdir)[0])

                if format == 'pkl':
                    with open(file_path, 'rb') as f:
                        loaded_data = pickle.load(f)
                    # For pickle, the nested structure should be preserved.
                    self.assertIsInstance(loaded_data[0]['run_data'].main_args, argparse.Namespace)
                    self.assertEqual(loaded_data[0]['run_data'].main_args.model, 'malinois')

                elif format == 'parquet':
                    loaded_df = pd.read_parquet(file_path)
                    
                    # Check column names for correct, deep flattening.
                    expected_cols = {
                        'exp_starttime_str',
                        'run_data:main_args:model',
                        'run_data:main_args:optimization',
                        'run_data:main_args:max_number_of_rounds',
                        'run_data:main_args:start_sequence',
                        'run_data:model_init_args',
                        'run_data:opt_init_args'
                    }
                    self.assertSetEqual(set(loaded_df.columns), expected_cols)
                    
                    # Check values.
                    self.assertEqual(loaded_df.loc[0, 'run_data:main_args:model'], 'malinois')
                    self.assertEqual(loaded_df.loc[0, 'run_data:main_args:max_number_of_rounds'], 1)
                    self.assertIsNone(loaded_df.loc[0, 'run_data:model_init_args'])
    
    def test_save_proposals_with_various_types(self):
        """
        Tests that `save_proposals` can correctly save various object types
        to both pickle and parquet formats.
        """
        args = argparse.Namespace(
            optimization='test_opt',
            model='test_model',
        )

        # Create a complex Namespace object to be nested in the dictionary.
        nested_args = argparse.Namespace(param_a='val_a', param_b=1.23)

        # Create a dictionary with a variety of types.
        write_dicts = [
            {
                'exp_starttime_str': '20231101-100000',
                'a_string': 'hello_world',
                'an_int': 99,
                'a_float': 3.14159,
                'a_numpy_array': np.array([10, 20, 30]),
                'a_namespace_obj': nested_args,
                'nested_dict': {
                    'nested_int': 42,
                    'nested_array': np.arange(2),
                }
            },
        ]

        for format in ['pkl', 'parquet']:
            with self.subTest(format=format):
                output_dir = os.path.join(self.test_dir.name, format)
                gcp_utils.save_proposals(write_dicts, args, output_dir, format=format)
                
                # Find the created file.
                expected_subdir = os.path.join(output_dir, 'test_opt_test_model', '20231101-100000')
                self.assertTrue(os.path.isdir(expected_subdir))
                files = os.listdir(expected_subdir)
                self.assertEqual(len(files), 1)
                file_path = os.path.join(expected_subdir, files[0])

                if format == 'pkl':
                    with open(file_path, 'rb') as f:
                        loaded_data = pickle.load(f)
                    
                    self.assertEqual(len(loaded_data), 1)
                    record = loaded_data[0]
                    self.assertEqual(record['a_string'], 'hello_world')
                    self.assertEqual(record['an_int'], 99)
                    self.assertIsInstance(record['a_namespace_obj'], argparse.Namespace)
                    self.assertEqual(record['a_namespace_obj'].param_a, 'val_a')
                    np.testing.assert_array_equal(record['a_numpy_array'], np.array([10, 20, 30]))
                    np.testing.assert_array_equal(record['nested_dict']['nested_array'], np.arange(2))

                elif format == 'parquet':
                    loaded_df = pd.read_parquet(file_path)
                    
                    self.assertEqual(len(loaded_df), 1)
                    record = loaded_df.iloc[0]
                    
                    # Check column names for correct flattening.
                    expected_cols = {
                        'exp_starttime_str', 'a_string', 'an_int', 'a_float', 
                        'a_numpy_array',
                        'a_namespace_obj:param_a', 'a_namespace_obj:param_b',
                        'nested_dict:nested_int', 'nested_dict:nested_array'
                    }
                    self.assertSetEqual(set(loaded_df.columns), expected_cols)
                    
                    # Check values and types.
                    self.assertEqual(record['a_string'], 'hello_world')
                    self.assertEqual(record['nested_dict:nested_int'], 42)
                    self.assertEqual(record['a_namespace_obj:param_a'], 'val_a')
                    self.assertEqual(record['a_namespace_obj:param_b'], 1.23)
                    np.testing.assert_array_equal(record['a_numpy_array'], np.array([10, 20, 30]))
                    np.testing.assert_array_equal(record['nested_dict:nested_array'], np.arange(2))

    def test_flatten_dicts_to_dataframe(self):
        # Test case 1: Simple, single-level nesting
        dicts1 = [{'a': {'b': 1}, 'c': 2}]
        expected1 = pd.DataFrame([{'a:b': 1, 'c': 2}])
        pd.testing.assert_frame_equal(gcp_utils._flatten_dicts_to_dataframe(dicts1), expected1)

        # Test case 2: Deeper nesting
        dicts2 = [{'a': {'b': {'c': 3}}, 'd': 4}]
        expected2 = pd.DataFrame([{'a:b:c': 3, 'd': 4}])
        pd.testing.assert_frame_equal(gcp_utils._flatten_dicts_to_dataframe(dicts2), expected2)

        # Test case 3: Multiple dictionaries
        dicts3 = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        expected3 = pd.DataFrame([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
        pd.testing.assert_frame_equal(gcp_utils._flatten_dicts_to_dataframe(dicts3), expected3)

        # Test case 4: Mixed structure with missing values
        dicts4 = [{'a': 1, 'b': {'c': 5}}, {'a': 2, 'b': {'d': 6}}]
        # When pandas creates a DataFrame from records with missing keys,
        # the column order is not guaranteed. We define the expected result
        # and then reorder its columns to match the actual result for robust testing.
        expected4 = pd.DataFrame([
            {'a': 1, 'b:c': 5.0, 'b:d': pd.NA},
            {'a': 2, 'b:c': pd.NA, 'b:d': 6.0}
        ])
        
        result4 = gcp_utils._flatten_dicts_to_dataframe(dicts4)
        
        # Ensure the expected DataFrame has the same column order.
        # We disable the dtype check here because pandas' type inference with
        # missing values can be inconsistent. The values are still checked.
        expected4 = expected4[result4.columns]
        pd.testing.assert_frame_equal(result4, expected4, check_dtype=False)

    def setUp(self):
        # Create a temporary directory for test outputs.
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # Clean up the temporary directory.
        self.test_dir.cleanup()

    def test_save_proposals_local(self):
        # Test saving proposals to a local file in both formats.
        args = argparse.Namespace(
            optimization='test_opt',
            model='test_model',
        )
        write_dicts = [
            {'exp_starttime_str': '20230101-120000', 'data': 'test1', 'nested': {'value': 1}},
            {'exp_starttime_str': '20230101-120000', 'data': 'test2', 'nested': {'value': 2}},
        ]
        
        # Test for both pkl and parquet formats
        for format in ['pkl', 'parquet']:
            with self.subTest(format=format):
                # Use a unique subdirectory for each format to avoid conflicts
                output_dir = os.path.join(self.test_dir.name, format)
                
                gcp_utils.save_proposals(write_dicts, args, output_dir, format=format)

                # Check that the file was created in the expected subdirectory
                expected_subdir = os.path.join(output_dir, 'test_opt_test_model', '20230101-120000')
                self.assertTrue(os.path.isdir(expected_subdir))
                
                files = os.listdir(expected_subdir)
                self.assertEqual(len(files), 1)
                self.assertTrue(files[0].endswith(f'.{format}'))

                # Check the content of the file
                file_path = os.path.join(expected_subdir, files[0])
                if format == 'pkl':
                    with open(file_path, 'rb') as f:
                        loaded_data = pickle.load(f)
                        self.assertEqual(len(loaded_data), 2)
                        self.assertEqual(loaded_data[0]['data'], 'test1')
                elif format == 'parquet':
                    loaded_df = pd.read_parquet(file_path)
                    self.assertEqual(len(loaded_df), 2)
                    self.assertEqual(loaded_df.loc[0, 'data'], 'test1')
                    self.assertEqual(loaded_df.loc[1, 'nested:value'], 2)

    def test_write_txt_file_local(self):
        # Test writing a simple text file locally.
        output_path = os.path.join(self.test_dir.name, 'test_output')
        content = "SUCCESS"
        
        gcp_utils.write_txt_file(output_path, content)
        
        expected_file = os.path.join(output_path, f'{content}.txt')
        self.assertTrue(os.path.exists(expected_file))
        
        with open(expected_file, 'r') as f:
            self.assertEqual(f.read(), content)

    @mock.patch('nucleobench.common.gcp_utils.get_role_client')
    def test_write_str_to_gcp(self, mock_get_role_client):
        # Test the function that writes a string to GCS by mocking the client provider.
        mock_client = mock.MagicMock()
        mock_get_role_client.return_value = mock_client
        
        mock_bucket = mock.MagicMock()
        mock_blob = mock.MagicMock()
        mock_file_context = io.StringIO()

        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.open.return_value.__enter__.return_value = mock_file_context

        gcs_path = 'gs://fake-bucket/fake/path/to/file.txt'
        content = 'hello world'
        
        gcp_utils.write_str_to_gcp(gcs_path, content, binary=False)
        
        mock_get_role_client.assert_called_once()
        mock_client.bucket.assert_called_once_with('fake-bucket')
        mock_bucket.blob.assert_called_once_with('fake/path/to/file.txt')
        mock_blob.open.assert_called_once_with('w')
        self.assertEqual(mock_file_context.getvalue(), content)

    @mock.patch('pandas.read_csv')
    def test_read_gcp_csv(self, mock_read_csv):
        # Test that the wrapper calls pd.read_csv with the correct GCS path.
        gcs_path = 'gs://fake-bucket/fake-dir/test.csv'
        expected_df = pd.DataFrame({'col1': [1, 2]})
        mock_read_csv.return_value = expected_df
        
        result_df = gcp_utils.read_gcp_csv(gcs_path)
        
        mock_read_csv.assert_called_once_with(gcs_path)
        pd.testing.assert_frame_equal(result_df, expected_df)

if __name__ == '__main__':
    unittest.main()