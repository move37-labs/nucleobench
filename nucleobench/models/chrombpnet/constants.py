"""Constants for ChromBPNet oracles.

HDMA fetal-tissue weights: Zenodo 15048278 (CC-BY 4.0).
https://zenodo.org/records/15048278

ENCODE K562 weights: HuggingFace kundajelab collection (ENCODE data-use policy).
https://huggingface.co/collections/kundajelab/encode-chrombpnet-models
Cite: Pampari et al. 2024.

Each model uses fold 0 of the bias-corrected (`chrombpnet_nobias`) model.
"""

import os

VOCAB_ = ["A", "C", "G", "T"]

SEQ_LEN = 2114

# ---------------------------------------------------------------------------
# HDMA fetal-tissue models (Zenodo)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ENCODE K562 models (HuggingFace)
# ---------------------------------------------------------------------------
# Naming pattern: kundajelab/encode-chrombpnet-{ASSAY}-{BIOSAMPLE}-{EXPERIMENT}-{ANNOTATION}
# Primary model filename confirmed from live HF repo tree; alternates use None so
# the loader resolves the path via the HF repo file listing at download time.

K562_MODELS_: dict[str, dict] = {
    "K562_ENCSR483RKN": {
        "repo_id": "kundajelab/encode-chrombpnet-ATAC-K562-ENCSR483RKN-ENCSR780QKO",
        "filename": "fold_0/model.chrombpnet_nobias.fold_0.ENCSR483RKN.h5",
    },
    "K562_ENCSR868FGK_v1": {
        "repo_id": "kundajelab/encode-chrombpnet-ATAC-K562-ENCSR868FGK-ENCSR467RSV",
        "filename": None,  # resolved at download time via repo listing
    },
    "K562_ENCSR868FGK_v2": {
        "repo_id": "kundajelab/encode-chrombpnet-ATAC-K562-ENCSR868FGK-ENCSR893SUD",
        "filename": None,
    },
}

K562_AVAILABLE_MODELS_: list[str] = list(K562_MODELS_)

ALL_AVAILABLE_MODELS_: list[str] = AVAILABLE_MODELS_ + K562_AVAILABLE_MODELS_
