"""Data loader for BPNet-ATAC start sequences.

Reads the 100 ATAC start sequences (each exactly 3,000 bp) from the in-repo
parquet at nucleobench/common/.cache_start_sequence_scores/bpnet_start_sequence_scores.parquet
(filtered to target_feature == "ATAC") and caches the result to
integration_tests/data_loaders/cache/bpnet_atac_start_sequences.parquet.

To use:
```python
from integration_tests.data_loaders import BPNetATACStartSequences
loader = BPNetATACStartSequences()
df = loader.get_data()  # DataFrame with a 'sequence' column, 100 rows
```
"""

from pathlib import Path

import pandas as pd

from .base import DataLoader

_REPO_ROOT = Path(__file__).parent.parent.parent
_SOURCE_PARQUET = (
    _REPO_ROOT
    / "nucleobench"
    / "common"
    / ".cache_start_sequence_scores"
    / "bpnet_start_sequence_scores.parquet"
)


class BPNetATACStartSequences(DataLoader):
    """Data loader for the 100 BPNet-ATAC start sequences (3,000 bp each).

    The source is the in-repo parquet at
    nucleobench/common/.cache_start_sequence_scores/bpnet_start_sequence_scores.parquet,
    filtered to target_feature == "ATAC".
    On first use it is read and cached to
    integration_tests/data_loaders/cache/bpnet_atac_start_sequences.parquet.
    """

    def _get_default_cache_path(self) -> Path:
        return Path(__file__).parent / "cache" / "bpnet_atac_start_sequences.parquet"

    def _download_and_process(self) -> pd.DataFrame:
        if not _SOURCE_PARQUET.exists():
            raise FileNotFoundError(
                f"Source parquet not found: {_SOURCE_PARQUET}\n"
                "Expected it at nucleobench/common/.cache_start_sequence_scores/"
                "bpnet_start_sequence_scores.parquet inside the repo."
            )
        df = pd.read_parquet(_SOURCE_PARQUET)
        df = df[df["target_feature"] == "ATAC"][["start_sequence"]].rename(
            columns={"start_sequence": "sequence"}
        )
        df = df.reset_index(drop=True)
        assert len(df) == 100, f"Expected 100 ATAC sequences, got {len(df)}"
        assert (df["sequence"].str.len() == 3_000).all(), (
            "Not all sequences are 3,000 bp"
        )
        return df
