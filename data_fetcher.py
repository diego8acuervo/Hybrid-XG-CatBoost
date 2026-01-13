"""
Core Data Fetching Module - Yahoo Finance Integration

Retrieves historical OHLCV data for all assets in the universe
with robust error handling and caching capabilities.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import logging
from typing import Dict, List, Optional
import pickle

logger = logging.getLogger(__name__)


class YahooFinanceFetcher:
    """
    Fetches historical price data from Yahoo Finance.
    
    Supports:
    - Batch data retrieval
    - Caching to avoid redundant API calls
    - Data validation
    - Missing value handling
    """
    
    def __init__(self, config: dict, cache_dir: str = "./data/raw"):
        """
        Initialize data fetcher.
        
        Args:
            config: Configuration dictionary with data settings
            cache_dir: Directory for caching downloaded data
        """
        self.config = config['data']
        self.cache_dir = cache_dir
        self.start_date = self.config.get('start_date', '2005-01-01')
        self.end_date = self.config.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        self.symbols = self.config.get('symbols', [])
        self.cache_enabled = self.config.get('cache_data', True)
        
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Data fetcher initialized - Period: {self.start_date} to {self.end_date}")
    
    def _get_cache_path(self, symbol: str) -> str:
        """Get cache file path for symbol."""
        return os.path.join(self.cache_dir, f"{symbol}_{self.start_date}_{self.end_date}.pkl")
    
    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from cache if exists."""
        if not self.cache_enabled:
            return None
        
        cache_path = self._get_cache_path(symbol)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logger.debug(f"Loaded {symbol} from cache")
                return data
            except Exception as e:
                logger.warning(f"Cache read failed for {symbol}: {e}")
                return None
        return None
    
    def _save_to_cache(self, symbol: str, data: pd.DataFrame) -> None:
        """Save data to cache."""
        if not self.cache_enabled:
            return
        
        try:
            cache_path = self._get_cache_path(symbol)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Cached {symbol}")
        except Exception as e:
            logger.warning(f"Cache write failed for {symbol}: {e}")
    
    def fetch_single(self, symbol: str, progress: bool = True) -> pd.DataFrame:
        """
        Fetch historical data for single symbol.
        
        Args:
            symbol: Ticker symbol
            progress: Show download progress
            
        Returns:
            DataFrame with OHLCV data
        """
        # Try cache first
        cached_data = self._load_from_cache(symbol)
        if cached_data is not None:
            return cached_data
        
        logger.info(f"Downloading {symbol}...")
        try:
            data = yf.download(
                symbol,
                start=self.start_date,
                end=self.end_date,
                progress=progress
            )
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            data['Symbol'] = symbol
            self._save_to_cache(symbol, data)
            logger.info(f"✓ Downloaded {symbol}: {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Failed to download {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_historical_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all symbols in universe.
        
        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        data = {}
        failed_symbols = []
        
        for symbol in self.symbols:
            df = self.fetch_single(symbol, progress=False)
            if df.empty:
                failed_symbols.append(symbol)
            else:
                data[symbol] = df
        
        if failed_symbols:
            logger.warning(f"Failed to fetch: {failed_symbols}")
        
        logger.info(f"✓ Successfully fetched {len(data)}/{len(self.symbols)} symbols")
        return data
    
    def fetch_combined(self) -> pd.DataFrame:
        """
        Fetch and combine data for all symbols into single DataFrame.
        
        Returns:
            DataFrame with 'Adj Close' prices for all symbols as columns
        """
        data = self.fetch_historical_data()
        
        combined = pd.DataFrame()
        for symbol, df in data.items():
            if 'Adj Close' in df.columns:
                combined[symbol] = df['Adj Close']
        
        combined = combined.sort_index()
        logger.info(f"Combined data shape: {combined.shape}")
        logger.info(f"Date range: {combined.index[0]} to {combined.index[-1]}")
        
        return combined
    
    def save_raw_data(self, data: Dict[str, pd.DataFrame], output_dir: str = "./data/raw") -> None:
        """Save raw data to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        for symbol, df in data.items():
            path = os.path.join(output_dir, f"{symbol}.csv")
            df.to_csv(path)
            logger.info(f"Saved {symbol} to {path}")


class CSVDataLoader:
    """Load data from local CSV files (alternative to Yahoo Finance)."""
    
    def __init__(self, data_dir: str = "./data/raw"):
        """Initialize CSV loader."""
        self.data_dir = data_dir
    
    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """Load CSV for single symbol."""
        path = os.path.join(self.data_dir, f"{symbol}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        logger.info(f"Loaded {symbol} from {path}")
        return df
    
    def load_all(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Load all symbols."""
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.load_symbol(symbol)
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")
        return data
