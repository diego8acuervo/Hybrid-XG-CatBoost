"""
Data Validator - Validate data quality and integrity

Performs checks on data completeness, consistency, and validity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate financial data quality and integrity."""
    
    def __init__(self, config: dict):
        """
        Initialize DataValidator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
    def validate_all(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Run all validation checks.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary of validation results
        """
        results = {
            'missing_values': self.check_missing_values(data),
            'date_gaps': self.check_date_gaps(data),
            'price_validity': self.check_price_validity(data),
            'volume_validity': self.check_volume_validity(data),
            'duplicates': self.check_duplicates(data)
        }
        
        return results
    
    def check_missing_values(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Check for missing values in data.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Dictionary with missing value statistics
        """
        missing = data.isnull().sum()
        missing_pct = (missing / len(data)) * 100
        
        result = {
            'count': missing.to_dict(),
            'percentage': missing_pct.to_dict(),
            'total_missing': int(missing.sum())
        }
        
        if result['total_missing'] > 0:
            logger.warning(f"Found {result['total_missing']} missing values")
        else:
            logger.info("✓ No missing values found")
            
        return result
    
    def check_date_gaps(self, data: pd.DataFrame, 
                        max_gap_days: int = 5) -> Dict[str, any]:
        """
        Check for gaps in date sequence.
        
        Args:
            data: DataFrame with datetime index
            max_gap_days: Maximum allowed gap in days
            
        Returns:
            Dictionary with gap information
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            return {'error': 'Index is not DatetimeIndex'}
        
        gaps = data.index.to_series().diff()
        large_gaps = gaps[gaps > pd.Timedelta(days=max_gap_days)]
        
        result = {
            'num_gaps': len(large_gaps),
            'gap_locations': large_gaps.index.tolist() if len(large_gaps) > 0 else []
        }
        
        if result['num_gaps'] > 0:
            logger.warning(f"Found {result['num_gaps']} date gaps > {max_gap_days} days")
        else:
            logger.info("✓ No significant date gaps found")
            
        return result
    
    def check_price_validity(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Check if prices are valid (positive, OHLC relationships).
        
        Args:
            data: DataFrame with OHLC columns
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        # Check for negative prices
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns:
                negative = (data[col] < 0).sum()
                if negative > 0:
                    issues.append(f"{col}: {negative} negative values")
        
        # Check OHLC relationships
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            high_violations = ((data['High'] < data['Low']) | 
                              (data['High'] < data['Open']) | 
                              (data['High'] < data['Close'])).sum()
            
            low_violations = ((data['Low'] > data['High']) | 
                             (data['Low'] > data['Open']) | 
                             (data['Low'] > data['Close'])).sum()
            
            if high_violations > 0:
                issues.append(f"High price violations: {high_violations}")
            if low_violations > 0:
                issues.append(f"Low price violations: {low_violations}")
        
        result = {
            'valid': len(issues) == 0,
            'issues': issues
        }
        
        if result['valid']:
            logger.info("✓ All prices are valid")
        else:
            logger.warning(f"Price validation issues: {issues}")
            
        return result
    
    def check_volume_validity(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Check if volume data is valid.
        
        Args:
            data: DataFrame with Volume column
            
        Returns:
            Dictionary with validation results
        """
        if 'Volume' not in data.columns:
            return {'valid': True, 'message': 'No volume column'}
        
        negative = (data['Volume'] < 0).sum()
        zero = (data['Volume'] == 0).sum()
        
        result = {
            'valid': negative == 0,
            'negative_count': int(negative),
            'zero_count': int(zero),
            'zero_percentage': float(zero / len(data) * 100)
        }
        
        if result['valid']:
            logger.info("✓ Volume data is valid")
        else:
            logger.warning(f"Found {negative} negative volume values")
            
        return result
    
    def check_duplicates(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Check for duplicate index values.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Dictionary with duplicate information
        """
        duplicates = data.index.duplicated().sum()
        
        result = {
            'count': int(duplicates),
            'has_duplicates': duplicates > 0
        }
        
        if result['has_duplicates']:
            logger.warning(f"Found {duplicates} duplicate index values")
        else:
            logger.info("✓ No duplicate indices found")
            
        return result
