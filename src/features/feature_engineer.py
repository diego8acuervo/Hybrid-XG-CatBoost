"""
Comprehensive Feature Engineering Pipeline

Implements all 178+ features as described in the paper:
- Time-series moments (volatility, skewness, kurtosis, entropy)
- Hurst exponent (multi-scale persistence)
- Cross-asset relationships (beta, correlation)
- Information-theoretic measures (KL divergence)
"""

import logging
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for systematic alpha model.
    
    Implements all feature categories from the research paper.
    """
    
    def __init__(self, config: dict):
        """Initialize feature engineer with configuration."""
        self.config = config.get('features', {})
        self.rolling_windows = self.config.get('rolling_windows', [21, 63])
        self.hurst_scales = self.config.get('hurst_scales', [16, 64, 256])
        self.entropy_bins = self.config.get('entropy_bins', 30)
        self.target_drawdown = self.config.get('target_drawdown', 0.01)
        self.target_horizon = self.config.get('target_horizon', 5)
        self.target_symbol = config.get('data', {}).get('target_symbol', 'BTC-USD')
    
    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute log returns from price series."""
        returns = np.log(prices / prices.shift(1))
        return returns.dropna()
    
    def compute_volatility(self, returns: pd.Series, window: int) -> pd.Series:
        """Rolling standard deviation."""
        return returns.rolling(window=window).std()
    
    def compute_skewness(self, returns: pd.Series, window: int) -> pd.Series:
        """Rolling skewness (3rd moment)."""
        return returns.rolling(window=window).apply(
            lambda x: skew(x, bias=True) if len(x) > 2 else np.nan
        )
    
    def compute_kurtosis(self, returns: pd.Series, window: int) -> pd.Series:
        """Rolling excess kurtosis (4th moment - 3)."""
        return returns.rolling(window=window).apply(
            lambda x: kurtosis(x, bias=True, fisher=True) if len(x) > 3 else np.nan
        )
    
    def compute_entropy(self, returns: pd.Series, window: int, bins: int = 30) -> pd.Series:
        """Rolling Shannon entropy of return distribution."""
        def _entropy(x):
            if len(x) < bins:
                return np.nan
            hist, _ = np.histogram(x, bins=bins)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist))
        
        return returns.rolling(window=window).apply(_entropy)
    
    def compute_hurst_exponent(self, returns: pd.Series, scale: int) -> pd.Series:
        """
        Compute Hurst exponent using rescaled range analysis.
        
        Higher Hurst (>0.5): trending/persistent
        Lower Hurst (<0.5): mean-reverting
        H=0.5: random walk
        
        Args:
            returns: Return series
            scale: Time scale for analysis (16, 64, 256)
            
        Returns:
            Series of Hurst exponents
        """
        def _hurst(x):
            if len(x) < scale * 2:
                return np.nan
            
            # Cumulative sum
            cumsum = np.cumsum(x - x.mean())
            
            # Range
            max_val = np.maximum.accumulate(cumsum)
            min_val = np.minimum.accumulate(cumsum)
            range_val = max_val - min_val
            
            # Standard deviation
            std_val = np.std(x, ddof=1)
            if std_val == 0:
                return np.nan
            
            # Rescaled range
            r_s = range_val / std_val
            
            # Hurst exponent (H = log(R/S) / log(N))
            n = np.arange(1, len(r_s) + 1)
            h = np.log(r_s) / np.log(n)
            return np.mean(h[-min(scale, len(h)):])
        
        return returns.rolling(window=scale * 2).apply(_hurst)
    
    def compute_rolling_beta(self, asset_returns: pd.Series,
                            market_returns: pd.Series, window: int) -> pd.Series:
        """Compute rolling beta vs market (target symbol).
        
        Beta = Cov(asset, market) / Var(market)
        Uses pandas optimized built-in methods for best performance.
        """
        # Beta = Covariance / Variance
        rolling_cov = asset_returns.rolling(window).cov(market_returns)
        rolling_var = market_returns.rolling(window).var()
        return rolling_cov / rolling_var
    
    def compute_rolling_correlation(self, asset_returns: pd.Series,
                                   market_returns: pd.Series, window: int) -> pd.Series:
        """Compute rolling correlation vs market."""
        return asset_returns.rolling(window=window).corr(market_returns)
    
    def compute_kl_divergence(self, returns: pd.Series, window_current: int,
                             window_reference: int, bins: int = 30) -> pd.Series:
        """
        Kullback-Leibler divergence vs reference distribution.
        
        Measures shift in return distribution (higher = more extreme).
        """
        def _kl_div(x):
            if len(x) < window_current:
                return np.nan
            
            current_returns = x[:window_current]
            ref_returns = x[window_current:]
            
            if len(ref_returns) < bins:
                return np.nan
            
            # Histograms
            hist_curr, edges = np.histogram(current_returns, bins=bins)
            hist_ref, _ = np.histogram(ref_returns, bins=edges)
            
            # Normalize
            p = hist_curr / hist_curr.sum()
            q = hist_ref / hist_ref.sum()
            
            # KL divergence with smoothing
            p = p + 1e-10
            q = q + 1e-10
            
            return np.sum(p * np.log(p / q))
        
        combined = pd.concat([returns] * 2, axis=1).T
        return returns.rolling(window=window_current + window_reference).apply(_kl_div)
    
    def create_target_variable(self, prices: pd.DataFrame) -> pd.Series:
        """
        Create binary target: 1 if 5-day target price drawdown >= 1%, else 0.

        y_t = 1 if sum(r_target[t+1:t+5]) <= -0.01, else 0
        """
        target_returns = self.compute_returns(prices[[self.target_symbol]])

        # Rolling sum of returns over horizon
        cumsum_returns = target_returns[self.target_symbol].rolling(
            window=self.target_horizon
        ).sum()
        
        # Shift forward to get future drawdown
        target = (cumsum_returns.shift(-self.target_horizon) <= -self.target_drawdown).astype(int)
        
        logger.info(f"Target variable created - Positive class: {target.sum()} samples")
        return target
    
    def engineer_all_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all 178+ features from raw price data.
        
        Args:
            prices: DataFrame with adjusted close prices for all symbols
            
        Returns:
            DataFrame with engineered features (rows: dates, cols: features)
        """
        logger.info("Starting feature engineering...")
        
        returns = self.compute_returns(prices)
        features = pd.DataFrame(index=returns.index)
        
        feature_count = 0
        
        # 1. TIME-SERIES MOMENTS (Statistical Features)
        logger.info("Computing time-series moments...")
        for symbol in prices.columns:
            if symbol not in returns.columns:
                continue
            
            r = returns[symbol]
            
            for window in self.rolling_windows:
                # Volatility
                vol_feat = self.compute_volatility(r, window)
                features[f'{symbol}_vol_{window}d'] = vol_feat
                feature_count += 1
                
                # Skewness
                skew_feat = self.compute_skewness(r, window)
                features[f'{symbol}_skew_{window}d'] = skew_feat
                feature_count += 1
                
                # Kurtosis
                kurt_feat = self.compute_kurtosis(r, window)
                features[f'{symbol}_kurtosis_{window}d'] = kurt_feat
                feature_count += 1
                
                # Entropy
                ent_feat = self.compute_entropy(r, window, self.entropy_bins)
                features[f'{symbol}_entropy_{window}d'] = ent_feat
                feature_count += 1
        
        # 2. HURST EXPONENT (Persistence / Mean-Reversion Indicator)
        logger.info("Computing Hurst exponents...")
        for symbol in prices.columns:
            if symbol not in returns.columns:
                continue
            
            r = returns[symbol]
            
            for scale in self.hurst_scales:
                hurst_feat = self.compute_hurst_exponent(r, scale)
                features[f'{symbol}_hurst_{scale}'] = hurst_feat
                feature_count += 1
        
        # 3. CROSS-ASSET RELATIONSHIPS (vs target symbol)
        logger.info(f"Computing cross-asset relationships vs {self.target_symbol}...")
        target_returns = returns[self.target_symbol]
        
        for symbol in prices.columns:
            if symbol not in returns.columns or symbol == self.target_symbol:
                continue
            
            asset_returns = returns[symbol]
            
            for window in self.rolling_windows:
                # Beta
                beta_feat = self.compute_rolling_beta(asset_returns, target_returns, window)
                features[f'{symbol}_beta_{self.target_symbol}_{window}d'] = beta_feat
                feature_count += 1
                
                # Correlation
                corr_feat = self.compute_rolling_correlation(asset_returns, target_returns, window)
                features[f'{symbol}_corr_{self.target_symbol}_{window}d'] = corr_feat
                feature_count += 1
        
        # 4. INFORMATION-THEORETIC MEASURES (KL Divergence)
        logger.info("Computing information-theoretic measures...")
        kl_window_pairs = [(21, 126), (63, 126)]  # current, reference window
        
        for symbol in prices.columns:
            if symbol not in returns.columns:
                continue
            
            r = returns[symbol]
            
            for w_curr, w_ref in kl_window_pairs:
                kl_feat = self.compute_kl_divergence(r, w_curr, w_ref, self.entropy_bins)
                features[f'{symbol}_kl_{w_curr}vs{w_ref}'] = kl_feat
                feature_count += 1
        
        logger.info(f"✓ Engineered {feature_count} features")
        logger.info(f"Feature matrix shape: {features.shape}")
        logger.info(f"Missing values before cleaning: {features.isna().sum().sum()}")
        
        # Remove columns with too many NaN values (>10%)
        nan_threshold = 0.1
        nan_pct = features.isna().mean()
        bad_features = nan_pct[nan_pct > nan_threshold].index
        if len(bad_features) > 0:
            logger.warning(f"Dropping {len(bad_features)} features with >{nan_threshold*100}% NaN values")
            features = features.drop(columns=bad_features)
        
        # Fill remaining NaN values: forward fill -> backward fill -> zero
        features = features.ffill().bfill().fillna(0)
        logger.info(f"Missing values after cleaning: {features.isna().sum().sum()}")
        logger.info(f"Final feature matrix shape: {features.shape}")
        
        return features
    
    def select_features_by_variance(self, features: pd.DataFrame,
                                   threshold: float = 1e-4) -> pd.DataFrame:
        """Remove low-variance features."""
        variances = features.var()
        low_var_features = variances[variances < threshold].index
        
        logger.info(f"Removing {len(low_var_features)} low-variance features")
        features = features.drop(columns=low_var_features)
        
        return features
    
    def select_features_by_correlation(self, features: pd.DataFrame,
                                      threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated feature pairs."""
        corr_matrix = features.corr().abs()
        
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        
        logger.info(f"Removing {len(to_drop)} highly correlated features")
        features = features.drop(columns=to_drop)
        
        return features
    
    def select_top_features_by_mi(self, features: pd.DataFrame, target: pd.Series,
                                 n_features: int = 80) -> Tuple[pd.DataFrame, Dict]:
        """
        Select top N features by mutual information with target.
        """
        from sklearn.feature_selection import mutual_info_classif
        
        logger.info("Computing mutual information scores...")
        
        # Align indices
        valid_idx = features.index.intersection(target.index)
        X = features.loc[valid_idx]
        y = target.loc[valid_idx]
        
        # Compute MI
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Rank features
        mi_ranking = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        selected_features = mi_ranking.head(n_features)
        logger.info(f"Top features by MI:\n{selected_features.head(13)}")
        
        # Select
        X_selected = X[selected_features.index]
        
        logger.info(f"Selected {n_features} features based on mutual information")
        
        return X_selected, mi_ranking.to_dict()
    
    def standardize_features(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Standardize features to mean=0, std=1."""
        if fit:
            self.feature_mean = features.mean()
            self.feature_std = features.std()
        
        features_std = (features - self.feature_mean) / (self.feature_std + 1e-10)
        logger.info("Features standardized")
        
        return features_std
