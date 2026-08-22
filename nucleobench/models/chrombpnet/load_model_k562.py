"""Fetch ENCODE K562 ChromBPNet nobias weights from HuggingFace and load via bpnet-lite.

Models live in the kundajelab/encode-chrombpnet collection:
  https://huggingface.co/collections/kundajelab/encode-chrombpnet-models

Naming pattern: kundajelab/encode-chrombpnet-{ASSAY}-{EXPERIMENT}-{ANNOTATION}

HuggingFace's own cache (~/.cache/huggingface/hub/) is used; no separate
NucleoBench cache directory is needed.  hf_hub_download is resumable and
thread-safe.

Cite: Pampari et al. 2024.

To test one locally:
```zsh
python -m nucleobench.models.chrombpnet.load_model_k562
```
"""

from bpnetlite import BPNet
from huggingface_hub import hf_hub_download, list_repo_tree

from nucleobench.models.bpnet.load_model import CountWrapper
from nucleobench.models.chrombpnet import constants as cb_constants


def _resolve_filename(repo_id: str) -> str:
    """Find fold_0/*nobias*.h5 in the HF repo tree.

    Used for alternate K562 entries whose exact filename has not been
    confirmed in advance.  Raises FileNotFoundError if no match is found.
    """
    candidates = [
        item.path
        for item in list_repo_tree(repo_id, recursive=True)
        if (
            item.path.startswith("fold_0/")
            and "nobias" in item.path
            and item.path.endswith(".h5")
        )
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No fold_0/*nobias*.h5 found in HuggingFace repo {repo_id!r}. "
            "Check the repo contents manually: "
            f"https://huggingface.co/{repo_id}/tree/main/fold_0"
        )
    # Prefer the first hit; all folds use the same nobias weights.
    return candidates[0]


def _get_h5_path(key: str) -> str:
    """Return the local HF-cached path to the fold-0 nobias .h5 for `key`."""
    cfg = cb_constants.K562_MODELS_[key]
    repo_id = cfg["repo_id"]
    filename = cfg["filename"] or _resolve_filename(repo_id)
    return hf_hub_download(repo_id=repo_id, filename=filename)


def download(key: str, override_weights_local_path: str | None = None):
    """Load a CountWrapper around the fold-0 nobias ChromBPNet for `key`."""
    if key not in cb_constants.K562_MODELS_:
        raise ValueError(
            f"Unknown K562 ChromBPNet key {key!r}. "
            f"Choose from {cb_constants.K562_AVAILABLE_MODELS_}"
        )

    if override_weights_local_path is not None:
        import os

        if not os.path.exists(override_weights_local_path):
            raise FileNotFoundError(override_weights_local_path)
        h5_path = override_weights_local_path
    else:
        h5_path = _get_h5_path(key)

    model = BPNet.from_chrombpnet(h5_path)
    return CountWrapper(model)


if __name__ == "__main__":
    download("K562_ENCSR483RKN")
