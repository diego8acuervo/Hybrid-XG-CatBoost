"""
Helper Utilities - Common utility functions

Collection of helper functions used across the project.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict
from pathlib import Path
import json


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, filepath: Union[str, Path]):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Union[str, Path]) -> dict:
    """
    Load JSON file to dictionary.
    
    Args:
        filepath: Input file path
        
    Returns:
        Dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_returns(prices: pd.Series, 
                     method: str = 'log') -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        method: 'log' for log returns, 'simple' for simple returns
        
    Returns:
        Returns series
    """
    if method == 'log':
        return np.log(prices / prices.shift(1))
    elif method == 'simple':
        return prices.pct_change()
    else:
        raise ValueError(f"Unknown method: {method}")


def annualize_metric(metric: float, 
                     periods_per_year: int = 252,
                     metric_type: str = 'return') -> float:
    """
    Annualize a metric.
    
    Args:
        metric: Metric value (return or volatility)
        periods_per_year: Number of periods in a year (252 for daily)
        metric_type: 'return' or 'volatility'
        
    Returns:
        Annualized metric
    """
    if metric_type == 'return':
        return (1 + metric) ** periods_per_year - 1
    elif metric_type == 'volatility':
        return metric * np.sqrt(periods_per_year)
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage string.
    
    Args:
        value: Numeric value (0.15 for 15%)
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{value * 100:.{decimals}f}%"


def chunks(lst: List, n: int):
    """
    Yield successive n-sized chunks from lst.
    
    Args:
        lst: List to chunk
        n: Chunk size
        
    Yields:
        Chunks of size n
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def merge_dicts(*dicts: dict) -> dict:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Variable number of dictionaries
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def safe_divide(numerator: float, denominator: float, 
                default: float = 0.0) -> float:
    """
    Safely divide two numbers, return default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except (ZeroDivisionError, TypeError):
        return default


def timestamp_to_str(timestamp: pd.Timestamp, 
                     format: str = '%Y-%m-%d') -> str:
    """
    Convert pandas Timestamp to string.
    
    Args:
        timestamp: Pandas Timestamp
        format: Date format string
        
    Returns:
        Formatted date string
    """
    return timestamp.strftime(format)
