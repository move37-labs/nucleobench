"""Load Malinois model.

Adapted from:
https://github.com/sjgosai/boda2/blob/main/boda/common/utils.py

Typical use:

```python
malinois_path = 'gs://tewhey-public-data/CODA_resources/malinois_artifacts__20211113_021200__287348.tar.gz'
my_model = load_model(malinois_path)
assert isinstance(my_model, torch.nn.Module)
```

To test this file:
```zsh
python -m nucleobench.models.malinois.load_model
```
"""

import os
import torch
import shutil
import tarfile
import tempfile

from typing import Optional

from google.cloud import storage

import nucleobench.models.malinois.model as _model

def load_model(artifact_path: str, has_cuda: Optional[bool] = None):
    
    if has_cuda is None:
        has_cuda = torch.cuda.device_count() >= 1

    with tempfile.TemporaryDirectory() as tmpdirname:
        unpack_artifact(artifact_path, download_path=tmpdirname)
        my_model = model_fn(os.path.join(tmpdirname, 'artifacts'))
        
    my_model.eval()
    if has_cuda:
        my_model.cuda()
    
    return my_model


def unpack_artifact(artifact_path, download_path='./'):
    """
    Unpack a tar archive artifact.

    Args:
        artifact_path (str): Path to the artifact.
        download_path (str, optional): Path to extract the artifact. Defaults to './'.
    """
    if artifact_path.startswith('gs://'):
        storage_client = storage.Client.create_anonymous_client()

        bucket = storage_client.bucket(artifact_path.split('/')[2])
        blob = bucket.blob('/'.join(artifact_path.split('/')[3:]))
        tar_model = os.path.join(download_path, os.path.basename(artifact_path))
        blob.download_to_filename(tar_model)
    else:
        assert os.path.isfile(artifact_path), "Could not find file at expected path."
        tar_model = artifact_path
        
    assert tarfile.is_tarfile(tar_model), f"Expected a tarfile at {tar_model}. Not found."
    
    shutil.unpack_archive(tar_model, download_path)
    print(f'archive unpacked in {download_path}',)

def model_fn(model_dir):
    """
    Load a model from a directory.

    Args:
        model_dir (str): Path to the model directory.

    Returns:
        torch.nn.Module: Loaded model in evaluation mode.
    """
    # The following line is needed because `weights_only=False` is required here.
    torch.serialization.add_safe_globals([_model.BassetBranched])
    checkpoint = torch.load(os.path.join(model_dir,'torch_checkpoint.pt'), weights_only=False)
    model_module = getattr(_model, checkpoint['model_module'])
    model        = model_module(**vars(checkpoint['model_hparams']))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded model from {checkpoint["timestamp"]} in eval mode')
    model.eval()
    return model


if __name__ == '__main__':
    # Test with real data or local data.
    #model = load_model('gs://tewhey-public-data/CODA_resources/malinois_artifacts__20211113_021200__287348.tar.gz')
    model = load_model('./malinois_artifacts__20211113_021200__287348.tar.gz')
    assert isinstance(model, torch.nn.Module)