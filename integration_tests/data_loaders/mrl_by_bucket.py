"""Data loader for MRL sequences by bucket (HIGH, MEDIUM, LOW)."""

import csv
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from Bio import Entrez, SeqIO

from .base import DataLoader


class MRLByBucket(DataLoader):
    """Data loader for ribosomal loading test sequences by expression bucket.
    
    Fetches sequences from NCBI and categorizes them by expression level:
    - HIGH: Housekeeping genes (ACTB, GAPDH, etc.)
    - MEDIUM: Regulated genes (TP53, MAPK1, etc.)
    - LOW: Repressed genes with uORFs (ATF4, DDIT3, etc.)
    """
    
    # NCBI requires an email address to use their API
    NCBI_EMAIL = "shor.joel@gmail.com"  # <--- PLEASE REPLACE WITH YOUR EMAIL
    
    SEQUENCE_LENGTH = 1022
    
    # The dataset: High (Housekeeping), Medium (Regulated), Low (Repressed/uORFs)
    GENE_TARGETS = [
        # --- HIGH BUCKET ---
        {"gene": "ACTB",     "accession": "NM_001101.5", "bucket": "HIGH"},
        {"gene": "GAPDH",    "accession": "NM_002046.7", "bucket": "HIGH"},
        {"gene": "EEF1A1",   "accession": "NM_001402.6", "bucket": "HIGH"},  # Elongation factor, extremely high expression
        {"gene": "RPL13A",   "accession": "NM_000977.4", "bucket": "HIGH"},  # Ribosomal protein, standard ref gene
        
        # --- MEDIUM BUCKET ---
        {"gene": "TP53",     "accession": "NM_000546.6", "bucket": "MEDIUM"},
        {"gene": "MAPK1",    "accession": "NM_002745.5", "bucket": "MEDIUM"}, # ERK2, standard signaling
        {"gene": "CTNNB1",   "accession": "NM_001904.4", "bucket": "MEDIUM"}, # Beta-catenin
        
        # --- LOW BUCKET ---
        {"gene": "ATF4",     "accession": "NM_001675.4", "bucket": "LOW"},
        {"gene": "DDIT3",    "accession": "NM_004083.6", "bucket": "LOW"},    # CHOP
        {"gene": "PPP1R15A", "accession": "NM_014330.5", "bucket": "LOW"},    # GADD34, has known uORFs
    ]
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the MRL by bucket data loader.
        
        Args:
            cache_dir: Optional directory for cache files. If None, uses default location
                      in integration_tests/data_loaders/cache/
        """
        super().__init__(cache_dir)
        Entrez.email = self.NCBI_EMAIL
    
    def _get_default_cache_path(self) -> Path:
        """Get the default cache file path.
        
        Returns:
            Path to the default cache CSV file
        """
        # Default cache location in the data_loaders directory
        default_dir = Path(__file__).parent / "cache"
        return default_dir / "mrl_by_bucket.csv"
    
    def _download_and_process(self) -> pd.DataFrame:
        """Download sequences from NCBI and process them.
        
        Returns:
            DataFrame with columns: gene, accession, bucket, sequence_length, sequence
        """
        print(f"Fetching {len(self.GENE_TARGETS)} sequences from NCBI...")
        
        results = []
        
        for target in self.GENE_TARGETS:
            acc_id = target["accession"]
            gene_name = target["gene"]
            bucket = target["bucket"]
            
            print(f"Downloading {gene_name} ({acc_id})...")
            
            try:
                # Fetch FASTA record from NCBI Nucleotide database
                handle = Entrez.efetch(db="nucleotide", id=acc_id, rettype="fasta", retmode="text")
                record = SeqIO.read(handle, "fasta")
                handle.close()
                
                # Extract sequence as string
                full_seq = str(record.seq)
                
                # Truncate to desired length
                truncated_seq = full_seq[:self.SEQUENCE_LENGTH]
                
                # Validation warning if sequence is too short
                if len(truncated_seq) < self.SEQUENCE_LENGTH:
                    print(f"  [WARNING] {gene_name} is shorter than {self.SEQUENCE_LENGTH} nt (Length: {len(truncated_seq)})")
                
                results.append({
                    'gene': gene_name,
                    'accession': acc_id,
                    'bucket': bucket,
                    'sequence_length': len(truncated_seq),
                    'sequence': truncated_seq
                })
                
                # Be polite to the NCBI server
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  [ERROR] Failed to fetch {gene_name}: {e}")
        
        df = pd.DataFrame(results)
        print(f"\nSuccess! Processed {len(df)} sequences")
        return df

