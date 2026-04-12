"""Data loader for ribosomal loading data from MTtrans repository.

Downloads data from the MTtrans GitHub repository evaluation testset results.
Supports multiple datasets: 3M, 3M_H, 3M_U, 3M_V.

To use:
```python
from integration_tests.data_loaders import RibosomalLoadingMTtransRepo
loader = RibosomalLoadingMTtransRepo(dataset_name='3M')
df = loader.get_data()  # Returns DataFrame with the data
```
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import DataLoader


# URLs for different MTtrans datasets
STRIDE_DATA_URLS = {
    '3M': 'https://raw.githubusercontent.com/holab-hku/MTtrans/main/evaluation/testset_result/stride1113_3M_pred.csv',
    '3M_H': 'https://raw.githubusercontent.com/holab-hku/MTtrans/main/evaluation/testset_result/stride11133m_MPA_H_pred.csv',
    '3M_U': 'https://raw.githubusercontent.com/holab-hku/MTtrans/main/evaluation/testset_result/stride11133m_MPA_U_pred.csv',
    '3M_V': 'https://raw.githubusercontent.com/holab-hku/MTtrans/main/evaluation/testset_result/stride11133m_MPA_V_pred.csv',
}


class RibosomalLoadingMTtransRepo(DataLoader):
    """Data loader for ribosomal loading data from MTtrans repository.
    
    Downloads evaluation testset results from the MTtrans GitHub repository.
    Supports multiple datasets:
    - 3M: Main 3M dataset
    - 3M_H: 3M dataset with H subset
    - 3M_U: 3M dataset with U subset
    - 3M_V: 3M dataset with V subset
    """
    
    def __init__(self, dataset_name: str, cache_dir: Optional[Path] = None):
        """Initialize the MTtrans ribosomal loading data loader.
        
        Args:
            dataset_name: Name of the dataset to load ('3M', '3M_H', '3M_U', '3M_V')
            cache_dir: Optional directory for cache files. If None, uses default location
                      in integration_tests/data_loaders/cache/
        """
        if dataset_name not in STRIDE_DATA_URLS:
            raise ValueError(f"Unknown dataset_name: {dataset_name}. Must be one of {list(STRIDE_DATA_URLS.keys())}")
        
        self.dataset_name = dataset_name
        super().__init__(cache_dir)
    
    def _get_default_cache_path(self) -> Path:
        """Get the default cache file path.
        
        Returns:
            Path to the default cache CSV file (named by dataset)
        """
        # Default cache location in the data_loaders directory
        default_dir = Path(__file__).parent / "cache"
        return default_dir / f"ribosomal_loading_mttrans_{self.dataset_name}.csv"
    
    def _download_real_data(self, url: str, path: Path) -> None:
        """Download data from URL using curl.
        
        Args:
            url: URL to download from
            path: Local path to save the file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['curl', '-L', url, '--output', str(path)], check=True)
    
    def _download_and_process(self) -> pd.DataFrame:
        """Download data from MTtrans repository and process it.
        
        Returns:
            DataFrame with the ribosomal loading data
        """
        url = STRIDE_DATA_URLS[self.dataset_name]
        print(f"Downloading MTtrans dataset '{self.dataset_name}' from {url}...")
        
        # Use temporary directory for download
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname) / f'{self.dataset_name}.csv'
            self._download_real_data(url, tmp_path)
            
            # Load the CSV
            data_df = pd.read_csv(tmp_path)
            
            assert len(data_df) > 0, f"Loaded DataFrame for {self.dataset_name} is empty"
            print(f"Success! Loaded {len(data_df)} rows for dataset '{self.dataset_name}'")
            
            return data_df

