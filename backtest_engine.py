"""
Walk-Forward Backtesting Framework

Implements proper time-series cross-validation for out-of-sample validation
with realistic trading execution and performance measurement.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class WalkForwardBacktest:
    """
    Walk-forward validation with rolling model retraining.
    
    Proper temporal validation avoiding look-ahead bias:
    - Train on historical data
    - Test on forward period
    - Rebalance model regularly
    - Track out-of-sample performance
    """
    
    def __init__(self, config: dict, ensemble, prices: pd.DataFrame,
                 features: pd.DataFrame, target: pd.Series):
        """
        Initialize backtest engine.
        
        Args:
            config: Configuration dictionary
            ensemble: Trained ensemble model
            prices: Price data for strategy execution
            features: Engineered features
            target: Target variable
        """
        self.config = config.get('walk_forward', {})
        self.backtest_config = config.get('backtest', {})
        self.prices = prices
        self.features = features
        self.target = target
        self.ensemble = ensemble
        
        self.train_size = self.config.get('train_size', 1000)
        self.test_size = self.config.get('test_size', 252)
        self.rebalance_freq = self.config.get('rebalance_frequency', 'weekly')
        
        self.results = []
        self.equity_curve = None
        self.trades = []
    
    def get_splits(self) -> List[Tuple[int, int]]:
        """Generate train-test split indices."""
        splits = []
        n_samples = len(self.features)
        
        train_idx = 0
        while train_idx + self.train_size + self.test_size <= n_samples:
            test_start = train_idx + self.train_size
            test_end = test_start + self.test_size
            
            splits.append((train_idx, test_start, test_end))
            
            # Rolling window
            train_idx += self.test_size
        
        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits
    
    def run(self) -> Dict:
        """
        Execute walk-forward backtest.
        
        Returns:
            Dictionary with performance metrics and equity curve
        """
        logger.info("=" * 60)
        logger.info("WALK-FORWARD BACKTEST")
        logger.info("=" * 60)
        
        splits = self.get_splits()
        if not splits:
            logger.error("Insufficient data for walk-forward splits")
            return {}
        
        predictions_oos = []
        actuals_oos = []
        
        for split_num, (train_start, test_start, test_end) in enumerate(splits):
            logger.info(f"\n[Split {split_num+1}/{len(splits)}] "
                       f"Train: [{train_start}:{test_start}], "
                       f"Test: [{test_start}:{test_end}]")
            
            # Extract train/test data
            X_train = self.features.iloc[train_start:test_start]
            y_train = self.target.iloc[train_start:test_start]
            X_test = self.features.iloc[test_start:test_end]
            y_test = self.target.iloc[test_start:test_end]
            
            # Train model on this window
            from src.models.ensemble_model import HybridEnsemble
            model = HybridEnsemble(self.ensemble.config)
            model.fit(X_train, y_train, validation_split=0.1)
            
            # Out-of-sample predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            predictions_oos.extend(y_pred_proba)
            actuals_oos.extend(y_test.values)
            
            # Store results
            self.results.append({
                'split': split_num,
                'train_start': train_start,
                'test_start': test_start,
                'test_end': test_end,
                'model': model,
                'y_pred': y_pred_proba,
                'y_actual': y_test.values,
                'dates': self.features.index[test_start:test_end]
            })
        
        # Overall out-of-sample performance
        from sklearn.metrics import (
            roc_auc_score, precision_score, recall_score, f1_score
        )
        
        predictions_oos = np.array(predictions_oos)
        actuals_oos = np.array(actuals_oos)
        
        oos_auc = roc_auc_score(actuals_oos, predictions_oos)
        y_pred_binary = (predictions_oos >= 0.5).astype(int)
        oos_precision = precision_score(actuals_oos, y_pred_binary)
        oos_recall = recall_score(actuals_oos, y_pred_binary)
        oos_f1 = f1_score(actuals_oos, y_pred_binary)
        
        logger.info(f"\nOut-of-Sample Performance:")
        logger.info(f"  ROC-AUC: {oos_auc:.4f}")
        logger.info(f"  Precision: {oos_precision:.4f}")
        logger.info(f"  Recall: {oos_recall:.4f}")
        logger.info(f"  F1-Score: {oos_f1:.4f}")
        
        return {
            'oos_auc': oos_auc,
            'oos_precision': oos_precision,
            'oos_recall': oos_recall,
            'oos_f1': oos_f1,
            'splits': self.results
        }
    
    def execute_strategy(self) -> pd.DataFrame:
        """
        Execute trading strategy based on model predictions.
        
        Strategy:
        - Long if P(crash) < 0.5
        - Short if P(crash) >= 0.5
        - Position size = confidence weighted
        """
        logger.info("\nExecuting strategy...")
        
        positions = []
        returns = []
        signals = []
        
        # Get SPY prices and returns
        spy_prices = self.prices['SPY'] if 'SPY' in self.prices.columns else self.prices.iloc[:, 0]
        spy_returns = np.log(spy_prices / spy_prices.shift(1))
        
        # Get ensemble predictions for all dates
        all_predictions = self.ensemble.predict_proba(self.features)[:, 1]
        
        for i in range(1, len(all_predictions)):
            pred_prob = all_predictions[i-1]
            
            # Position sizing based on confidence
            confidence = abs(pred_prob - 0.5) * 2  # Range [0, 1]
            
            if pred_prob < 0.5:
                # Low crash risk: LONG
                position = confidence * 1.0
                signal = 1
            else:
                # High crash risk: SHORT
                position = confidence * -1.0
                signal = -1
            
            positions.append(position)
            signals.append(signal)
            
            # Strategy return = position * market return
            market_return = spy_returns.iloc[i]
            strat_return = position * market_return
            returns.append(strat_return)
        
        # Create results dataframe
        strategy_df = pd.DataFrame({
            'Date': self.features.index[1:],
            'Position': positions,
            'Signal': signals,
            'Prediction': all_predictions[:-1],
            'Strategy_Return': returns,
            'SPY_Return': spy_returns.iloc[1:].values,
            'SPY_Price': spy_prices.iloc[1:].values
        })
        
        strategy_df['Strategy_Cumulative'] = (1 + strategy_df['Strategy_Return']).cumprod() * 100
        strategy_df['SPY_Cumulative'] = (1 + strategy_df['SPY_Return']).cumprod() * 100
        
        self.strategy_results = strategy_df
        
        logger.info("✓ Strategy execution complete")
        return strategy_df
    
    def compute_performance_metrics(self) -> Dict:
        """Compute comprehensive performance metrics."""
        if self.strategy_results is None:
            logger.warning("Strategy not executed yet")
            return {}
        
        strategy_ret = self.strategy_results['Strategy_Return']
        spy_ret = self.strategy_results['SPY_Return']
        
        # Annualization factor (252 trading days)
        periods = len(strategy_ret)
        years = periods / 252
        
        # Returns
        total_return_strat = (1 + strategy_ret).prod() - 1
        total_return_spy = (1 + spy_ret).prod() - 1
        annual_return_strat = (1 + total_return_strat) ** (1/years) - 1
        annual_return_spy = (1 + total_return_spy) ** (1/years) - 1
        
        # Volatility
        annual_vol_strat = strategy_ret.std() * np.sqrt(252)
        annual_vol_spy = spy_ret.std() * np.sqrt(252)
        
        # Sharpe (assuming 0% risk-free rate)
        sharpe_strat = annual_return_strat / annual_vol_strat if annual_vol_strat > 0 else 0
        sharpe_spy = annual_return_spy / annual_vol_spy if annual_vol_spy > 0 else 0
        
        # Drawdown
        cumret_strat = (1 + strategy_ret).cumprod()
        cumret_spy = (1 + spy_ret).cumprod()
        
        running_max_strat = cumret_strat.cummax()
        drawdown_strat = (cumret_strat - running_max_strat) / running_max_strat
        max_dd_strat = drawdown_strat.min()
        
        running_max_spy = cumret_spy.cummax()
        drawdown_spy = (cumret_spy - running_max_spy) / running_max_spy
        max_dd_spy = drawdown_spy.min()
        
        # Win rate
        win_rate = (strategy_ret > 0).sum() / len(strategy_ret)
        
        # Information ratio (vs SPY)
        excess_ret = strategy_ret - spy_ret
        info_ratio = excess_ret.mean() / excess_ret.std() * np.sqrt(252)
        
        # CAPM Alpha and Beta (OLS regression)
        from scipy import stats
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            spy_ret, strategy_ret
        )
        
        beta = slope
        daily_alpha = intercept
        annual_alpha = daily_alpha * 252
        alpha_t_stat = daily_alpha / std_err * np.sqrt(252)
        
        metrics = {
            'Strategy': {
                'Total Return': total_return_strat,
                'Annual Return': annual_return_strat,
                'Annual Volatility': annual_vol_strat,
                'Sharpe Ratio': sharpe_strat,
                'Max Drawdown': max_dd_strat,
                'Win Rate': win_rate
            },
            'SPY': {
                'Total Return': total_return_spy,
                'Annual Return': annual_return_spy,
                'Annual Volatility': annual_vol_spy,
                'Sharpe Ratio': sharpe_spy,
                'Max Drawdown': max_dd_spy
            },
            'Relative': {
                'Information Ratio': info_ratio,
                'CAPM Beta': beta,
                'CAPM Alpha (Annual)': annual_alpha,
                'Alpha T-stat': alpha_t_stat,
                'Excess Return': excess_ret.mean() * 252
            }
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("STRATEGY PERFORMANCE METRICS")
        logger.info("=" * 60)
        logger.info(f"\nStrategy:")
        logger.info(f"  Annual Return: {annual_return_strat*100:.2f}%")
        logger.info(f"  Volatility: {annual_vol_strat*100:.2f}%")
        logger.info(f"  Sharpe Ratio: {sharpe_strat:.2f}")
        logger.info(f"  Max Drawdown: {max_dd_strat*100:.2f}%")
        logger.info(f"  Win Rate: {win_rate*100:.1f}%")
        
        logger.info(f"\nSPY Benchmark:")
        logger.info(f"  Annual Return: {annual_return_spy*100:.2f}%")
        logger.info(f"  Volatility: {annual_vol_spy*100:.2f}%")
        logger.info(f"  Sharpe Ratio: {sharpe_spy:.2f}")
        logger.info(f"  Max Drawdown: {max_dd_spy*100:.2f}%")
        
        logger.info(f"\nRelative Metrics:")
        logger.info(f"  Information Ratio: {info_ratio:.2f}")
        logger.info(f"  Beta: {beta:.2f}")
        logger.info(f"  Alpha (annual): {annual_alpha*100:.2f}%")
        logger.info(f"  Alpha T-stat: {alpha_t_stat:.2f}")
        
        return metrics
