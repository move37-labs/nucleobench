"""Data loader for Enformer start sequences.

Reads 100 real genomic sequences (each exactly 196,608 bp) from the in-repo
parquet at nucleobench/common/.cache_start_sequence_scores/enformer_start_sequence_scores.parquet
and caches the result to integration_tests/data_loaders/cache/enformer_start_sequences.parquet.

To use:
```python
from integration_tests.data_loaders import EnformerStartSequences
loader = EnformerStartSequences()
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
    / "enformer_start_sequence_scores.parquet"
)


class EnformerStartSequences(DataLoader):
    """Data loader for the 100 Enformer start sequences (196,608 bp each).

    The source is the in-repo parquet at
    nucleobench/common/.cache_start_sequence_scores/enformer_start_sequence_scores.parquet.
    On first use it is read and cached to
    integration_tests/data_loaders/cache/enformer_start_sequences.parquet.
    """

    def _get_default_cache_path(self) -> Path:
        return Path(__file__).parent / "cache" / "enformer_start_sequences.parquet"

    def _download_and_process(self) -> pd.DataFrame:
        if not _SOURCE_PARQUET.exists():
            raise FileNotFoundError(
                f"Source parquet not found: {_SOURCE_PARQUET}\n"
                "Expected it at nucleobench/common/.cache_start_sequence_scores/"
                "enformer_start_sequence_scores.parquet inside the repo."
            )
        df = pd.read_parquet(_SOURCE_PARQUET)
        df = df[["start_sequence"]].rename(columns={"start_sequence": "sequence"})
        assert len(df) == 100, f"Expected 100 sequences, got {len(df)}"
        assert (df["sequence"].str.len() == 196_608).all(), (
            "Not all sequences are 196,608 bp"
        )
        return df
