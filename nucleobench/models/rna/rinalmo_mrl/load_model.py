"""Load RinAlmo model finetuned on mean ribosomal loading.


To test this file:
```zsh
python -m nucleobench.models.rna.rinalmo_mrl.load_model
```
"""

import os
import subprocess
import tempfile

import torch
from google.cloud import storage

from . import constants
from .rinalmo.ribosome_loading import RibosomeLoadingPredictionWrapper


def load_model(ft_wts_url: str = constants.FT_ZENODO_URL_,
               has_cuda: bool | None = None,
               force_download: bool = False,
               ) -> RibosomeLoadingPredictionWrapper:
    if has_cuda is None:
        has_cuda = torch.cuda.device_count() >= 1

    my_model = RibosomeLoadingPredictionWrapper(force_cpu=not has_cuda)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Download finetuned weights.
        if not force_download and os.path.exists(constants.CACHE_FT_):
            # Use cached weights.
            ft_wts_path = constants.CACHE_FT_
        elif os.path.exists(ft_wts_url):
            # Use local weights.
            ft_wts_path = ft_wts_url
        else:
            # Download weights.
            ft_wts_path = os.path.join(tmpdirname, 'rinalmo_giga_mrl_ft.pt')
            download_artifact(ft_wts_url, ft_wts_path)

        # Load weights into relevant parts of model.
        wts = torch.load(ft_wts_path, weights_only=True)
        if not has_cuda:
            wts = modify_flash_attn_wts_for_cpu(wts)
        # Use strict=False because inv_freq buffers are initialized, not loaded
        missing, unexpected = my_model.load_state_dict(wts, strict=False)
        # Verify ONLY inv_freq is missing (this is expected and correct)
        non_inv_freq_missing = [k for k in missing if 'inv_freq' not in k]
        if non_inv_freq_missing:
            raise ValueError(f"Unexpected missing keys: {non_inv_freq_missing}")
        if unexpected:
            raise ValueError(f"Unexpected keys in checkpoint: {unexpected}")

    my_model.eval()
    if has_cuda:
        my_model.cuda()

    return my_model


def modify_flash_attn_wts_for_cpu(wts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Modify flash attention weights for use on CPU.

    This means manually carving out regular attention from flash attention weights.

    General rules:
    - Map ex `lm.transformer.blocks.0.mh_attn.Wqkv.weight` ->
        'lm.transformer.blocks.0.mh_attn.mh_attn.to_q.weight',
        'lm.transformer.blocks.0.mh_attn.mh_attn.to_k.weight',
        'lm.transformer.blocks.0.mh_attn.mh_attn.to_v.weight',
    - Map ex `lm.transformer.blocks.0.mh_attn.out_proj.weight` ->
        'lm.transformer.blocks.0.mh_attn.mh_attn.out_proj.weight',
    """
    wts_keys = list(wts.keys())
    for k in wts_keys:
        if 'mh_attn.Wqkv.weight' in k:
            # Carve out 3 weights, and delete the flash attention weight.
            assert wts[k].ndim == 2
            assert wts[k].shape[0] % 3 == 0, wts[k].shape

            q_wts_name = k.replace('mh_attn.Wqkv.weight', 'mh_attn.mh_attn.to_q.weight')
            k_wts_name = k.replace('mh_attn.Wqkv.weight', 'mh_attn.mh_attn.to_k.weight')
            v_wts_name = k.replace('mh_attn.Wqkv.weight', 'mh_attn.mh_attn.to_v.weight')

            wts[q_wts_name], wts[k_wts_name], wts[v_wts_name]  = wts[k].chunk(3, dim=0)

            del wts[k]
        elif 'mh_attn.out_proj.weight' in k:
            # Map to the regular attention output projection weight.
            out_proj_wts_name = k.replace('mh_attn.out_proj.weight', 'mh_attn.mh_attn.out_proj.weight')
            wts[out_proj_wts_name] = wts[k]
            del wts[k]

    return wts



def download_artifact(url: str, output_path='./'):
    """Download an artifact from Zenodo or GCS."""
    print(f"Downloading artifact from {url} to {output_path}")
    if url.startswith('gs://'):
        # Use GCP storage client for GCS downloads
        storage_client = storage.Client()
        bucket = storage_client.bucket(url.split('/')[2])
        blob = bucket.blob('/'.join(url.split('/')[3:]))
        blob.download_to_filename(output_path)
    else:
        # Use curl for HTTP/HTTPS downloads (e.g., Zenodo)
        subprocess.run(['curl', url, '--output', output_path])
    assert os.path.exists(output_path), f"File {output_path} does not exist from {url} after download."
    return output_path


if __name__ == '__main__':
    #load_model()  # If the weights are in the cache, checks that they load quickly.
    load_model(force_download=True)  # Should take about 20 minutes to download and load.
