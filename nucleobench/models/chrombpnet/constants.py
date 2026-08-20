"""Constants for HDMA ChromBPNet oracles.

Weights: Zenodo 15048278 (HDMA ChromBPNet models - Part 1), CC-BY 4.0.
https://zenodo.org/records/15048278

Each tarball contains 5 folds. The oracle uses fold 0 of the bias-corrected
(`chrombpnet_nobias`) model, per HDMA's recommended downstream path.
"""

import os

VOCAB_ = ["A", "C", "G", "T"]

SEQ_LEN = 2114

ZENODO_RECORD = "https://zenodo.org/records/15048278"

# Confirmed from Zenodo API (94 *.gz files, excluding bundle_1_MD5SUMS.tsv).
# Internal path confirmed from HDMA docs and a partial Adrenal_c0 tar listing.
NOBIAS_RELPATH = "{key}/{key}__fold_0__chrombpnet_nobias.h5"

# Cluster counts per tissue (c0 .. c{n-1}), matching Zenodo Part 1 keys.
_TISSUE_COUNTS = {
    "Adrenal": 4,
    "Brain": 18,
    "Eye": 19,
    "Heart": 16,
    "Liver": 14,
    "Lung": 18,
    "Muscle": 5,
}

AVAILABLE_MODELS_ = [
    f"{tissue}_c{i}" for tissue, n in _TISSUE_COUNTS.items() for i in range(n)
]


def cache_dir() -> str:
    root = os.environ.get(
        "NUCLEOBENCH_CACHE_DIR", os.path.expanduser("~/.cache/nucleobench")
    )
    return os.path.join(root, "chrombpnet")


def cache_path(key: str) -> str:
    return os.path.join(cache_dir(), f"{key}.h5")
