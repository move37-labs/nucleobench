"""Utils for reading from and writing to GCP.

To test this with the cloud:
```zsh
python -m nucleobench.common.gcp_utils
```
"""

from typing import Any, Generator

import argparse
import copy
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
    return os.path.join(base_dir, f'{opt_method}_{model}', exp_start_time, f'{timestamp}.pkl')

def save_proposals(
        write_dicts: list[dict],
        args: argparse.Namespace, 
        output_path: str,
        ):
    """
    Save the proposals and associated arguments to a file.

    This function saves the generated proposals and the input arguments used for their generation
    to a file. The file is named based on the current timestamp and a random tag. The saved file can
    be placed in a local directory or on Google Cloud Storage if a 'gs://' path is provided in the arguments.

    Args:
        write_dicts: A list of dictionary of things to write.
        args: Args for the job. Best to keep them close to the output.
        output_path: Directory to write the output to, either locally or on GCP. Format of output is:
            {output_path}/{opt_method}_{model}/{exp start time}/{reults of exp N}.pkl

    Returns:
        None
    """
    def _tensor2np(obj):
        return obj.detach().clone().numpy() if isinstance(obj, torch.Tensor) else obj
    args = {k: _tensor2np(v) for k, v in vars(args).items()}
    write_dicts = [{k: _tensor2np(v) for k, v in x.items()} for x in write_dicts]
    save_dicts = copy.deepcopy(write_dicts)

    filename = get_filepath(
        base_dir=output_path,
        opt_method=args['optimization'],
        model=args['model'],
        exp_start_time=save_dicts[0]['exp_starttime_str'],
        timestamp=time.strftime("%Y%m%d_%H%M%S"),
    )
    
    # TODO(joelshor): Used to save as torch tensors. Now, saving as a pickled dictionary
    # of numpy arrays. Eventually, save as PyArrow arrays to parquet.
    if filename.startswith('gs://'):
        bucket_name = filename.split('/')[2]
        write_str_to_gcp(
            gcs_output_path=filename,
            content=pickle.dumps(save_dicts),
            binary=True,
            bucket_name=bucket_name,
        )
    else:
        try:
            if os.path.dirname(filename) != '' and os.path.dirname(filename) != '.':
                os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(save_dicts, f)
        except PermissionError as e:
            raise PermissionError(f'Permission error: {os.path.dirname(filename)}, {filename}') from e
    
    print(f'Proposals deposited at:\n\t{filename}')


def get_role_client(service_json_path: str = constants.SERVICE_KEY_FILE_LOCATION):
    try:
        gcp_client = storage.Client.from_service_account_json(service_json_path)
    except FileNotFoundError as e:
        raise ValueError(
            'GCP service key not found. You are probably trying to write output to a private '
            'bucket. To write to your own bucket, change the bucket name and service key '
            f'credentials in `nucleobench/common/constants.py`: {service_json_path}') from e
    return gcp_client

def write_str_to_gcp(
        gcs_output_path: str,
        content: Any,
        binary: bool,
        bucket_name: str = constants.GCP_OUTPUT_BUCKET_NAME,
        ):
    assert gcs_output_path.startswith('gs://'), 'gcs_output_path must be a GCS path.'
    gcs_output_path = gcs_output_path[len('gs://'):]
    bucket_name, blob_fn = gcs_output_path.split('/', 1)

    # Instantiates a client.
    storage_client = get_role_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_fn)

    write_type = 'wb' if binary else 'w'
    with blob.open(write_type) as f:
        f.write(content)
        
        
def download_gcp_folder_to_local(dir: str, local_dir: str, bucket: str = constants.GCP_OUTPUT_BUCKET_NAME):
    subprocess.call(['mkdir', '-p', local_dir])
    subprocess.call(['gcloud', 'storage', 'cp', '-r', f'gs://{bucket}/{dir}', local_dir])


def list_files_recursively(local_dir: str) -> Generator[str, None, None]:
    """Recursively lists all files in a given directory."""
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            yield os.path.join(root, file)


if __name__ == '__main__':
    args = {
        'fake_arg': 'fake_value',
        'fake_arg2': 'fake_value2',
        'optimization': 'dummy',
        'model': 'dummy_model',
    }
    proposal1 = {
        'string': 'ACTC',
        'energy': 1.0,
    }
    proposal2 = {
        'string': 'ACTC',
        'energy': 2.0,
    }

    write_dict = {
        'proposals': [proposal1, proposal2],
        'exp_starttime_str': '102010',
    }

    save_proposals([write_dict, write_dict], argparse.Namespace(**args), './gcp_utils_test/dummy')
    save_proposals([write_dict, write_dict], argparse.Namespace(**args), f'gs://{constants.GCP_OUTPUT_BUCKET_NAME}/gcp_utils_test/dummy')