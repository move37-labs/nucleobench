"""Fetch HDMA ChromBPNet nobias weights and load them via bpnet-lite.

Each Zenodo file is a ~470 MB gzipped tar. Only fold-0 `chrombpnet_nobias.h5`
is cached (~small); the rest of the archive is discarded.

https://zenodo.org/records/15048278

To test one locally:
```zsh
python -m nucleobench.models.chrombpnet.load_model
```
"""

import os
import shutil
import subprocess
import tempfile

from bpnetlite import BPNet

from nucleobench.models.bpnet.load_model import CountWrapper
from nucleobench.models.chrombpnet import constants as cb_constants


def get_url(key: str) -> str:
    if key not in cb_constants.AVAILABLE_MODELS_:
        raise ValueError(
            f"Unknown ChromBPNet cell type {key!r}. "
            f"Choose from {cb_constants.AVAILABLE_MODELS_}"
        )
    return f"{cb_constants.ZENODO_RECORD}/files/{key}.gz?download=1"


def _ensure_cached_h5(key: str) -> str:
    """Return the local path to `{key}.h5`, downloading from Zenodo if needed."""
    dest = cb_constants.cache_path(key)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest

    os.makedirs(cb_constants.cache_dir(), exist_ok=True)
    url = get_url(key)
    nobias_rel = cb_constants.NOBIAS_RELPATH.format(key=key)

    with tempfile.TemporaryDirectory() as tmpdirname:
        tar_path = os.path.join(tmpdirname, f"{key}.gz")
        subprocess.run(
            [
                "curl",
                "-fL",
                "-C",
                "-",
                "--retry",
                "5",
                "--retry-delay",
                "5",
                url,
                "--output",
                tar_path,
            ],
            check=True,
        )
        subprocess.run(
            ["tar", "-xzf", tar_path, "-C", tmpdirname],
            check=True,
        )
        src = os.path.join(tmpdirname, nobias_rel)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Expected nobias weights at {nobias_rel} inside {key}.gz, "
                f"not found after extract."
            )
        tmp_dest = dest + ".tmp"
        shutil.copy2(src, tmp_dest)
        os.replace(tmp_dest, dest)

    return dest


def download(key: str, override_weights_local_path: str | None = None):
    """Load a CountWrapper around the fold-0 nobias ChromBPNet for `key`."""
    if override_weights_local_path is not None:
        h5_path = override_weights_local_path
        if not os.path.exists(h5_path):
            raise FileNotFoundError(h5_path)
    else:
        h5_path = _ensure_cached_h5(key)

    model = BPNet.from_chrombpnet(h5_path)
    return CountWrapper(model)


if __name__ == "__main__":
    download("Adrenal_c0")
