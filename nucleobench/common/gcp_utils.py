"""Utils for reading from and writing to GCP.

To test this with the cloud:
```zsh
python -m nucleobench.common.gcp_utils
```
"""

from typing import Any, Generator

import argparse
import pandas as pd
import pyarrow
import os
import pickle
import subprocess
import torch
import time
from google.cloud import storage

from nucleobench.common import constants


def get_filepath(
    base_dir: str,
    opt_method: str,
    model: str,
    exp_start_time: str,
    timestamp: str,
    ) -> str:
    return os.path.join(base_dir, f'{opt_method}_{model}', exp_start_time, f'{timestamp}')


def _write_dicts_to_save_dicts(write_dicts: list[dict]) -> list[dict]:
    def _tensor2np(obj):
        return obj.detach().clone().numpy() if isinstance(obj, torch.Tensor) else obj
    return [{k: _tensor2np(v) for k, v in x.items()} for x in write_dicts]


def _flatten_dicts_to_dataframe(write_dicts: list[dict]) -> pd.DataFrame:
    """
    Flattens a list of complex, potentially nested objects into a pandas DataFrame.

    It handles nested dictionaries, argparse.Namespace objects, and other
    custom objects with a `__dict__` or `_asdict` attribute (like namedtuples).

    Nested keys are concatenated with a ':' separator. For example:
    {'a': {'b': 1}} becomes a column named 'a:b'.
    {'c': Namespace(d=2)} becomes a column named 'c:d'.

    Args:
        write_dicts: A list of dictionaries to flatten.

    Returns:
        A pandas DataFrame representing the flattened data.
    """
    def _flatten_recursive(obj: Any, parent_key: str = '', sep: str = ':') -> dict:
        """Recursively flattens a dictionary-like object."""
        items = []

        # Convert the object to a dictionary if possible
        if hasattr(obj, '_asdict'): # Handle namedtuples
            obj = obj._asdict()
        elif hasattr(obj, '__dict__'): # Handle Namespace and other custom objects
            obj = vars(obj)

        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.extend(_flatten_recursive(v, new_key, sep).items())
        else:
            # It's a primitive value, so append it.
            items.append((parent_key, obj))
        
        return dict(items)

    flattened_records = [_flatten_recursive(d) for d in write_dicts]
    return pd.DataFrame(flattened_records)


def save_proposals(
    write_dicts: list[dict],
    args: argparse.Namespace,
    output_path: str,
    format: str,
    ):
    """
    Save proposals and associated arguments to a file in either Parquet or Pickle format.

    Args:
        write_dicts: A list of dictionaries of things to write.
        args: Args for the job, containing metadata like optimization and model name.
        output_path: Directory to write the output to, either locally or on GCP.
        format: The format to save the file in, either 'parquet' or 'pkl'.
    """
    if format not in ['parquet', 'pkl']:
        raise ValueError(f"Unsupported format: '{format}'. Must be 'parquet' or 'pkl'.")

    save_dicts = _write_dicts_to_save_dicts(write_dicts)
    
    base_filename = get_filepath(
        base_dir=output_path,
        opt_method=args.optimization,
        model=args.model,
        exp_start_time=save_dicts[0]['exp_starttime_str'],
        timestamp=time.strftime("%Y%m%d_%H%M%S"),
    )
    filename = f'{base_filename}.{format}'
    
    is_gcs = filename.startswith('gs://')

    if format == 'parquet':
        try:
            data_df = _flatten_dicts_to_dataframe(save_dicts)
            if is_gcs:
                    data_df.to_parquet(filename)
            else:
                dir_name = os.path.dirname(filename)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                with open(filename, 'wb') as f:
                    data_df.to_parquet(f)
        except pyarrow.lib.ArrowInvalid as e:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(data_df)
            #raise ValueError(data_df.head(1)) from e
            raise e
    elif format == 'pkl':
        if is_gcs:
            write_str_to_gcp(gcs_output_path=filename, content=pickle.dumps(save_dicts), binary=True)
        else:
            dir_name = os.path.dirname(filename)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(save_dicts, f)
    else:
        raise ValueError(f"Unsupported format: '{format}'. Must be 'parquet' or 'pkl'.")
    
    print(f'Proposals deposited at:\n\t{filename}')


def get_role_client(service_json_path: str = constants.SERVICE_KEY_FILE_LOCATION):
    try:
        gcp_client = storage.Client.from_service_account_json(service_json_path)
    except (ValueError, TypeError, FileNotFoundError):
        gcp_client = storage.Client()  # When run in the Cloud.
    return gcp_client


def _parse_gcp_output_path(gcs_output_path: str) -> tuple[str, str]:
    assert gcs_output_path.startswith('gs://'), 'gcs_output_path must be a GCS path.'
    gcs_output_path = gcs_output_path[len('gs://'):]
    bucket_name, blob_fn = gcs_output_path.split('/', 1)
    return bucket_name, blob_fn


def write_str_to_gcp(
    gcs_output_path: str,
    content: Any,
    binary: bool,
    bucket_name: str = constants.GCP_OUTPUT_BUCKET_NAME,
    ):
    bucket_name, blob_fn = _parse_gcp_output_path(gcs_output_path)

    # Instantiates a client.
    storage_client = get_role_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_fn)

    write_type = 'wb' if binary else 'w'
    with blob.open(write_type) as f:
        f.write(content)


def list_files_recursively(local_dir: str) -> Generator[str, None, None]:
    """Recursively lists all files in a given directory."""
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            yield os.path.join(root, file)
            

def write_txt_file(output_path: str, content: str):
    """Write a ex. 'SUCCESS.txt' file."""
    if output_path.startswith('gs://'):
        write_str_to_gcp(
            gcs_output_path=os.path.join(output_path, f'{content}.txt'),
            content=content,
            binary=False,
            bucket_name=output_path.split('/')[2],
        )
    else:
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, f'{content}.txt'), 'w') as f:
            f.write(content)
            
            
def read_gcp_csv(gcs_path: str) -> pd.DataFrame:
    """Read a CSV file from a GCS path."""
    return pd.read_csv(gcs_path)