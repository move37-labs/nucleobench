"""Abstract base class for data loaders."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import pandas as pd


class DataLoader(ABC):
    """Abstract base class for data loaders.
    
    Subclasses should implement:
    - `_get_default_cache_path()`: Return the default cache file path
    - `_download_and_process()`: Download and process data, return DataFrame
    - Optionally override `_get_cache_path()` for custom cache location logic
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the data loader.
        
        Args:
            cache_dir: Optional directory for cache files. If None, uses default location.
        """
        self.cache_dir = cache_dir
    
    def _get_cache_path(self) -> Path:
        """Get the cache file path.
        
        Returns:
            Path to the cache file (CSV or Parquet)
        """
        if self.cache_dir is not None:
            return self.cache_dir / self._get_default_cache_path().name
        return self._get_default_cache_path()
    
    @abstractmethod
    def _get_default_cache_path(self) -> Path:
        """Get the default cache file path.
        
        Returns:
            Path to the default cache file location
        """
        pass
    
    @abstractmethod
    def _download_and_process(self) -> pd.DataFrame:
        """Download and process data from source.
        
        Returns:
            DataFrame with the processed data
        """
        pass
    
    def populate_cache(self) -> Path:
        """Download and process data, then save to cache.
        
        Returns:
            Path to the cache file
        """
        cache_path = self._get_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading and processing data...")
        df = self._download_and_process()
        
        # Save to cache (use parquet if path ends with .parquet, gzip if .gz, otherwise CSV)
        if cache_path.suffix == '.parquet':
            df.to_parquet(cache_path, index=False)
        elif cache_path.suffix == '.gz' or cache_path.suffixes[-1] == '.gz':
            # Save as gzipped CSV
            df.to_csv(cache_path, index=False, compression='gzip')
        else:
            df.to_csv(cache_path, index=False)
        
        print(f"Data cached to: {cache_path}")
        return cache_path
    
    def get_data(self) -> pd.DataFrame:
        """Get data, loading from cache if available, otherwise downloading.
        
        Returns:
            DataFrame with the data
        """
        cache_path = self._get_cache_path()
        
        # Check if cache exists
        if cache_path.exists():
            print(f"Loading data from cache: {cache_path}")
            if cache_path.suffix == '.parquet':
                return pd.read_parquet(cache_path)
            elif cache_path.suffix == '.gz' or cache_path.suffixes[-1] == '.gz':
                # Handle gzipped CSV files
                return pd.read_csv(cache_path, compression='gzip')
            else:
                return pd.read_csv(cache_path)
        
        # Cache doesn't exist, download and process
        print(f"Cache not found at {cache_path}, downloading...")
        self.populate_cache()
        
        # Load from the newly created cache
        if cache_path.suffix == '.parquet':
            return pd.read_parquet(cache_path)
        elif cache_path.suffix == '.gz' or cache_path.suffixes[-1] == '.gz':
            # Handle gzipped CSV files
            return pd.read_csv(cache_path, compression='gzip')
        else:
            return pd.read_csv(cache_path)

