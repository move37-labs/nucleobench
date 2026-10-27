"""Data loader for mouse ES-cell gene expression sequences by bucket.

Extracts sequences for Pou5f1, Nanog, Alb and a gene desert from mm10.

Assume mm10.fa is in the cache directory or will be downloaded.

The sequences are TSS-centered and categorized by expression in mouse ES cells:
- Pou5f1 (Oct4): ES master regulator, high in E14 and Bruce4
- Nanog: ES marker, high in E14 and Bruce4
- Alb (Albumin): liver-specific, silent in ES cells
- Gene Desert: no annotated gene in the predicted window

Coordinates are mm10 ncbiRefSeqCurated transcript bounds; all three genes are
on the + strand. `gene_start`/`gene_end` are offsets into the extracted
sequence, so that callers can score the gene body directly.

To use:
```python
from integration_tests.data_loaders import MouseEsExpressionByBucket
loader = MouseEsExpressionByBucket()
df = loader.get_data()  # Returns DataFrame with sequences
```
"""

import gzip
import shutil
from pathlib import Path

import pandas as pd
import pyfaidx
import requests
from tqdm import tqdm

from .base import DataLoader

# Transcript bounds, mm10 ncbiRefSeqCurated. All on the + strand.
# name: (chromosome, tx_start, tx_end, bucket)
GENES = {
    "POU5F1": ("chr17", 35_506_031, 35_510_777, "HIGH"),  # NM_013633.3
    "NANOG": ("chr6", 122_707_488, 122_714_633, "HIGH"),  # NM_001289831.1
    "ALB": ("chr5", 90_460_869, 90_476_602, "LOW"),  # NM_009654.4
}
# No annotated gene within the central 196,608 bp that Borzoi predicts.
GENE_DESERT_CHROM = "chr2"
GENE_DESERT_START = 21_990_000
GENE_DESERT_END = 22_010_000

SEQ_LENGTHS_TO_EXTRACT = [524_288]

MM10_URL = "http://hgdownload.cse.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz"


class MouseEsExpressionByBucket(DataLoader):
    """Data loader for mouse ES-cell gene expression sequences by bucket.

    Sequences are extracted at 524,288 bp (for Borzoi), centered on each
    gene's transcription start site.
    """

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the mouse ES gene expression data loader.

        Args:
            cache_dir: Optional directory for cache files. If None, uses default location
                      in integration_tests/data_loaders/cache/
        """
        super().__init__(cache_dir)

    def _get_default_cache_path(self) -> Path:
        """Get the default cache file path.

        Returns:
            Path to the default cache Parquet file
        """
        default_dir = Path(__file__).parent / "cache"
        return default_dir / "mouse_es_expression_by_bucket.parquet"

    def _get_mm10_path(self) -> Path:
        """Get the path to the mm10.fa file.

        Returns:
            Path to mm10.fa (will be in cache directory)
        """
        cache_dir = self._get_cache_path().parent
        return cache_dir / "mm10.fa"

    def _download_mm10(self, download_path: Path) -> None:
        """Downloads and decompresses mm10.fa.gz.

        Args:
            download_path: Path where mm10.fa should be saved
        """
        download_path.parent.mkdir(parents=True, exist_ok=True)
        gz_path = download_path.with_suffix(".fa.gz")

        print(f"Downloading mm10.fa.gz from {MM10_URL}...")
        response = requests.get(MM10_URL, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with (
            open(gz_path, "wb") as f,
            tqdm(
                desc="mm10.fa.gz",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    size = f.write(chunk)
                    bar.update(size)

        print("Decompressing mm10.fa.gz...")
        with gzip.open(gz_path, "rb") as f_in:
            with open(download_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        gz_path.unlink()
        print(f"mm10.fa saved to {download_path}")

    def _extract_window(
        self, fasta_handle, chrom: str, center: int, length: int
    ) -> tuple[str, int]:
        """Extracts a sequence of a given length centered at a location.

        Args:
            fasta_handle: Pyfaidx Fasta handle
            chrom: Chromosome name (e.g., 'chr17')
            center: Center position for extraction
            length: Desired sequence length

        Returns:
            (sequence, window_start), where window_start is the zero-based
            genomic coordinate of the sequence's first base.
        """
        window_start = center - (length // 2)
        sequence = fasta_handle[chrom][window_start : window_start + length].seq.upper()
        assert len(sequence) == length, (
            f"Expected length {length}, but got {len(sequence)}"
        )
        return sequence, window_start

    def _download_and_process(self) -> pd.DataFrame:
        """Download mm10 if needed and extract sequences.

        Returns:
            DataFrame with columns: gene, bucket, chromosome, center,
            sequence_length, gene_start, gene_end, sequence
        """
        mm10_path = self._get_mm10_path()

        if not mm10_path.exists():
            print(f"mm10.fa not found at {mm10_path}, downloading...")
            self._download_mm10(mm10_path)

        print(f"Opening FASTA file: {mm10_path}")
        fasta = pyfaidx.Fasta(str(mm10_path))

        targets = dict(GENES)
        targets["GENE_DESERT"] = (
            GENE_DESERT_CHROM,
            GENE_DESERT_START,
            GENE_DESERT_END,
            "LOW",
        )

        results = []
        for seq_length in SEQ_LENGTHS_TO_EXTRACT:
            print(f"\nExtracting sequences of length {seq_length}...")
            for gene, (chrom, tx_start, tx_end, bucket) in targets.items():
                sequence, window_start = self._extract_window(
                    fasta, chrom, tx_start, seq_length
                )
                results.append(
                    {
                        "gene": gene,
                        "bucket": bucket,
                        "chromosome": chrom,
                        "center": tx_start,
                        "sequence_length": seq_length,
                        # Offsets of the gene body within `sequence`.
                        "gene_start": tx_start - window_start,
                        "gene_end": tx_end - window_start,
                        "sequence": sequence,
                    }
                )

        df = pd.DataFrame(results)
        print(f"\nSuccess! Extracted {len(df)} sequences")
        return df
