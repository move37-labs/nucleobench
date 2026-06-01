"""Data loader for muscle gene expression sequences by bucket.

Extracts sequences for CKM, ALB, and a gene desert region from hg38.

Assume hg38.fa is in the cache directory or will be downloaded.

The sequences are extracted from specific genomic locations:
- CKM (Creatine Kinase Muscle): High expression in muscle
- ALB (Albumin): High expression in liver
- Gene Desert: Low/no expression region

To use:
```python
from integration_tests.data_loaders import MuscleGeneExpressionByBucket
loader = MuscleGeneExpressionByBucket()
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

# Define genomic coordinates
CKM_CHROM = "chr19"
CKM_TSS = 45322875

ALB_CHROM = "chr4"
ALB_TSS = 73404287

GENE_DESERT_CHROM = "chr8"
GENE_DESERT_CENTER = 127150000

SEQ_LENGTHS_TO_EXTRACT = [196_608, 524_288]

HG38_URL = "http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"


class MuscleGeneExpressionByBucket(DataLoader):
    """Data loader for muscle gene expression sequences by bucket.

    Extracts sequences for CKM, ALB, and a gene desert region from hg38.
    The sequences are categorized by expression level:
    - CKM: High expression in muscle
    - ALB: High expression in liver
    - Gene Desert: Low/no expression region

    Sequences are extracted at two different lengths:
    - 196,608 bp (for Enformer)
    - 524,288 bp (for Borzoi)
    """

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the muscle gene expression data loader.

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
        # Default cache location in the data_loaders directory
        default_dir = Path(__file__).parent / "cache"
        return default_dir / "muscle_gene_expression_by_bucket.parquet"

    def _get_hg38_path(self) -> Path:
        """Get the path to the hg38.fa file.

        Returns:
            Path to hg38.fa (will be in cache directory)
        """
        cache_dir = self._get_cache_path().parent
        return cache_dir / "hg38.fa"

    def _download_hg38(self, download_path: Path) -> None:
        """Downloads and decompresses hg38.fa.gz.

        Args:
            download_path: Path where hg38.fa should be saved
        """
        download_path.parent.mkdir(parents=True, exist_ok=True)
        gz_path = download_path.with_suffix(".fa.gz")

        print(f"Downloading hg38.fa.gz from {HG38_URL}...")
        response = requests.get(HG38_URL, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with (
            open(gz_path, "wb") as f,
            tqdm(
                desc="hg38.fa.gz",
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

        print("Decompressing hg38.fa.gz...")
        with gzip.open(gz_path, "rb") as f_in:
            with open(download_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        gz_path.unlink()
        print(f"hg38.fa saved to {download_path}")

    def _extract_sequence(
        self, fasta_handle, chrom: str, center: int, length: int, style: str
    ) -> str:
        """Extracts a sequence of a given length centered at a specific location.

        Args:
            fasta_handle: Pyfaidx Fasta handle
            chrom: Chromosome name (e.g., 'chr19')
            center: Center position for extraction
            length: Desired sequence length
            style: Extraction style - 'enformer' or 'borzoi'

        Returns:
            Extracted DNA sequence as string
        """
        if style == "enformer":
            start = center - (length // 2)
            end = center + ((length - 1) // 2)
        elif style == "borzoi":
            half_l = length // 2
            start = center - (half_l - 1)
            end = center + half_l
        else:
            raise ValueError(f"Unknown style: {style}")

        sequence = fasta_handle[chrom][start - 1 : end].seq.upper()
        assert len(sequence) == length, (
            f"Expected length {length}, but got {len(sequence)}"
        )
        return sequence

    def _download_and_process(self) -> pd.DataFrame:
        """Download hg38 if needed and extract sequences.

        Returns:
            DataFrame with columns: gene, chromosome, center, sequence_length, style, sequence
        """
        hg38_path = self._get_hg38_path()

        # Download hg38 if it doesn't exist
        if not hg38_path.exists():
            print(f"hg38.fa not found at {hg38_path}, downloading...")
            self._download_hg38(hg38_path)

        # Open FASTA file
        print(f"Opening FASTA file: {hg38_path}")
        fasta = pyfaidx.Fasta(str(hg38_path))

        # Extract sequences
        results = []

        for seq_length in SEQ_LENGTHS_TO_EXTRACT:
            print(f"\nExtracting sequences of length {seq_length}...")

            # Determine style based on length
            style = "enformer" if seq_length == 196_608 else "borzoi"

            # Extract CKM sequence (high expression in muscle)
            ckm_sequence = self._extract_sequence(
                fasta, CKM_CHROM, CKM_TSS, seq_length, style
            )
            results.append(
                {
                    "gene": "CKM",
                    "bucket": "HIGH",
                    "chromosome": CKM_CHROM,
                    "center": CKM_TSS,
                    "sequence_length": seq_length,
                    "style": style,
                    "sequence": ckm_sequence,
                }
            )

            # Extract ALB sequence (high expression in liver)
            alb_sequence = self._extract_sequence(
                fasta, ALB_CHROM, ALB_TSS, seq_length, style
            )
            results.append(
                {
                    "gene": "ALB",
                    "bucket": "HIGH",
                    "chromosome": ALB_CHROM,
                    "center": ALB_TSS,
                    "sequence_length": seq_length,
                    "style": style,
                    "sequence": alb_sequence,
                }
            )

            # Extract Gene Desert sequence (low/no expression)
            gene_desert_sequence = self._extract_sequence(
                fasta, GENE_DESERT_CHROM, GENE_DESERT_CENTER, seq_length, style
            )
            results.append(
                {
                    "gene": "GENE_DESERT",
                    "bucket": "LOW",
                    "chromosome": GENE_DESERT_CHROM,
                    "center": GENE_DESERT_CENTER,
                    "sequence_length": seq_length,
                    "style": style,
                    "sequence": gene_desert_sequence,
                }
            )

        df = pd.DataFrame(results)
        print(f"\nSuccess! Extracted {len(df)} sequences")
        return df
