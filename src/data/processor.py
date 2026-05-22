"""
Data Processor - Data preprocessing and cleaning utilities

Handles data quality checks, missing value imputation, and preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import logging
# Check if boto3 is installed, if not, install
try:
    import boto3
    boto3_available = True
except ImportError:
    boto3_available = False
    # If boto3 is not available, provide instructions for installation
    print("boto3 is not installed. Please install it using 'pip install boto3'.")
    

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and clean raw financial data."""
    
    def __init__(self, config: dict):
        """
        Initialize DataProcessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data: handle missing values, outliers, etc.
        
        Args:
            data: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        df = data.copy()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Forward fill missing values (common for financial data)
        df = df.fillna(method='ffill')
        
        # Remove remaining NaNs at start
        df = df.dropna()
        
        logger.info(f"Data cleaned: {df.shape}")
        return df
    
    def handle_missing_values(self, data: pd.DataFrame, 
                             method: str = 'ffill') -> pd.DataFrame:
        """
        Handle missing values in data.
        
        Args:
            data: DataFrame with potential missing values
            method: 'ffill', 'bfill', 'interpolate', or 'drop'
            
        Returns:
            DataFrame with handled missing values
        """
        if method == 'ffill':
            return data.fillna(method='ffill')
        elif method == 'bfill':
            return data.fillna(method='bfill')
        elif method == 'interpolate':
            return data.interpolate(method='linear')
        elif method == 'drop':
            return data.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def remove_outliers(self, data: pd.DataFrame, 
                       n_std: float = 5.0) -> pd.DataFrame:
        """
        Remove outliers beyond n standard deviations.
        
        Args:
            data: DataFrame
            n_std: Number of standard deviations for threshold
            
        Returns:
            DataFrame with outliers removed
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            mean = data[col].mean()
            std = data[col].std()
            lower = mean - n_std * std
            upper = mean + n_std * std
            
            # Replace outliers with NaN, then forward fill
            data.loc[(data[col] < lower) | (data[col] > upper), col] = np.nan
        
        data = data.fillna(method='ffill')
        return data
    
    def resample_data(self, data: pd.DataFrame, 
                     freq: str = 'D') -> pd.DataFrame:
        """
        Resample data to different frequency.
        
        Args:
            data: DataFrame with datetime index
            freq: Target frequency ('D', 'W', 'M', etc.)
            
        Returns:
            Resampled DataFrame
        """
        return data.resample(freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    
    def align_data(self, *dataframes: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Align multiple DataFrames to common index.
        
        Args:
            *dataframes: Variable number of DataFrames
            
        Returns:
            List of aligned DataFrames
        """
        common_idx = dataframes[0].index
        for df in dataframes[1:]:
            common_idx = common_idx.intersection(df.index)
        
        return [df.loc[common_idx] for df in dataframes]


    def loadS3(self, bucket: str, key: str) -> pd.DataFrame:
        """
        Load data from S3 bucket.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            
        Returns:
            DataFrame with loaded data
        """
        s3_client = boto3.client("s3")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(response["Body"])
        return df
