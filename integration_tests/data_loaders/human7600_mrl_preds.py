"""Data loader for human7600 MRL predictions dataset.

Downloads the human7600_mrl_preds.csv file from GitHub user attachments associated with RinAlmo.
This dataset contains 7600 sequences with their MRL predictions and targets.

To use:
```python
from integration_tests.data_loaders import Human7600MRLPreds
loader = Human7600MRLPreds()
df = loader.get_data()  # Returns DataFrame with columns: sequence, mrl_predicted, mrl_target
```
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from .base import DataLoader


class Human7600MRLPreds(DataLoader):
    """Data loader for human7600 MRL predictions dataset.
    
    Downloads the human7600_mrl_preds.csv file which contains:
    - sequence: The RNA sequence
    - mrl_predicted: Predicted MRL value
    - mrl_target: Target MRL value
    
    The dataset contains 7600 sequences total.
    """
    
    # URL for the dataset
    DATA_URL = 'https://github.com/user-attachments/files/23168017/human7600_mrl_preds.csv'
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the human7600 MRL predictions data loader.
        
        Args:
            cache_dir: Optional directory for cache files. If None, uses default location
                      in integration_tests/data_loaders/cache/
        """
        super().__init__(cache_dir)
    
    def _get_default_cache_path(self) -> Path:
        """Get the default cache file path.
        
        Returns:
            Path to the cache file (CSV format)
        """
        return Path(__file__).parent / "cache" / "human7600_mrl_preds.csv"
    
    def _download_and_process(self) -> pd.DataFrame:
        """Download and process the human7600 MRL predictions dataset.
        
        Returns:
            DataFrame with columns: sequence, mrl_predicted, mrl_target
        """
        print(f"Downloading human7600 MRL predictions from {self.DATA_URL}...")
        
        # Download directly using pandas (it can handle URLs)
        df = pd.read_csv(self.DATA_URL)
        
        print(f"Success! Loaded {len(df)} rows")
        return df

