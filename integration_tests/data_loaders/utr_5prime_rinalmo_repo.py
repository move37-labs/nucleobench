"""Data loader for 5' UTR data from RiNALMo repository."""

import tarfile
import tempfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from .base import DataLoader

MEGABYTE = 1_000_000

# URL for downloading Human 5'UTR library from NCBI GEO (GSE114002)
HUMAN_5UTR_LIB_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE114002&format=file"


class UTR5PrimeRinalmoRepo(DataLoader):
    """Data loader for synthetic Human 5'UTR library from RiNALMo repository.

    Downloads data from NCBI GEO (GSE114002) and extracts the CSV file.
    The data contains 5' UTR sequences with varying lengths (25-100 nt)
    and their corresponding ribosome loading values.
    """

    EXPECTED_FILENAME = "GSM4084997_varying_length_25to100.csv.gz"

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the 5' UTR RiNALMo repo data loader.

        Args:
            cache_dir: Optional directory for cache files. If None, uses default location
                      in integration_tests/data_loaders/cache/
        """
        super().__init__(cache_dir)

    def _get_default_cache_path(self) -> Path:
        """Get the default cache file path.

        Returns:
            Path to the default cache CSV file
        """
        # Default cache location in the data_loaders directory
        default_dir = Path(__file__).parent / "cache"
        return default_dir / "utr_5prime_rinalmo_repo.csv.gz"

    def _get_remote_data_url(self) -> str:
        """Get the remote data URL.

        Returns:
            URL string for downloading the data
        """
        return HUMAN_5UTR_LIB_URL

    def _write_and_update_progress_bar(self, file, progress, chunk):
        """Helper to write chunk and update progress bar."""
        progress.update(len(chunk) / MEGABYTE)
        file.write(chunk)

    def _download_archive(self, url: str, local_dir_path: Path) -> Path:
        """Download archive from URL.

        Args:
            url: URL to download from
            local_dir_path: Directory to save the archive

        Returns:
            Path to the downloaded archive
        """
        filename = url.split("/")[-1]
        local_file_path = local_dir_path / filename

        print(f"Downloading archive from {url}...")
        r = requests.get(url, stream=True)

        with open(local_file_path, 'wb') as f:
            with tqdm(total=int(r.headers['Content-Length']) / MEGABYTE, unit="MB") as progress_bar:
                for chunk in r.iter_content(chunk_size=1024):
                    self._write_and_update_progress_bar(f, progress_bar, chunk)

        return local_file_path

    def _download_and_process(self) -> pd.DataFrame:
        """Download data from RiNALMo repository and process it.

        Returns:
            DataFrame with the 5' UTR data
        """
        url = self._get_remote_data_url()

        # Use temporary directory for download and extraction
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)

            # Download archive
            archive_path = self._download_archive(url, tmpdir)

            # Extract archive
            print("Extracting archive...")
            with tarfile.open(archive_path, 'r') as tar:
                tar.extractall(tmpdir)

            # Find the expected CSV file
            csv_path = tmpdir / self.EXPECTED_FILENAME
            assert csv_path.exists(), \
                f"Expected file {self.EXPECTED_FILENAME} not found in extracted archive"

            # Load the CSV
            print(f"Loading data from {csv_path}...")
            data_df = pd.read_csv(csv_path)

            assert len(data_df) > 0, "Loaded DataFrame is empty"
            print(f"Success! Loaded {len(data_df)} rows")

            return data_df

