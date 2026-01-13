"""
Main Entry Point - Systematic Alpha Generation Pipeline

Execute full pipeline: data fetch → feature engineering → model training → backtesting
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alpha_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def stage_data_fetch(config: dict) -> pd.DataFrame:
    """Stage 1: Download data from Yahoo Finance."""
    logger.info("\n" + "="*60)
    logger.info("STAGE 1: DATA FETCHING")
    logger.info("="*60)
    
    from src.data.fetcher import YahooFinanceFetcher
    
    fetcher = YahooFinanceFetcher(config)
    data = fetcher.fetch_combined()
    
    # Save raw data
    os.makedirs('data/raw', exist_ok=True)
    data.to_csv('data/raw/prices_combined.csv')
    logger.info(f"✓ Data saved: {data.shape}")
    
    return data


def stage_feature_engineering(config: dict, prices: pd.DataFrame) -> tuple:
    """Stage 2: Engineer features and create target variable."""
    logger.info("\n" + "="*60)
    logger.info("STAGE 2: FEATURE ENGINEERING")
    logger.info("="*60)
    
    from src.features.feature_engineer import FeatureEngineer
    
    engineer = FeatureEngineer(config)
    
    # Engineer features
    features = engineer.engineer_all_features(prices)
    
    # Create target
    target = engineer.create_target_variable(prices)
    
    # Filter features
    features = engineer.select_features_by_variance(features)
    features = engineer.select_features_by_correlation(features)
    
    # Select top features by mutual information
    features, mi_scores = engineer.select_top_features_by_mi(
        features, target,
        n_features=config['features'].get('top_features', 80)
    )
    
    # Standardize
    features = engineer.standardize_features(features, fit=True)
    
    # Save
    os.makedirs('data/processed', exist_ok=True)
    features.to_csv('data/processed/features.csv')
    target.to_csv('data/processed/target.csv')
    
    logger.info(f"✓ Features saved: {features.shape}")
    logger.info(f"✓ Target saved: {target.shape}")
    
    return features, target


def stage_model_training(config: dict, features: pd.DataFrame,
                        target: pd.Series):
    """Stage 3: Train ensemble model."""
    logger.info("\n" + "="*60)
    logger.info("STAGE 3: MODEL TRAINING")
    logger.info("="*60)
    
    from src.models.ensemble_model import HybridEnsemble
    
    # Align indices
    valid_idx = features.index.intersection(target.index)
    X = features.loc[valid_idx].dropna()
    y = target.loc[X.index]
    
    logger.info(f"Training data: {X.shape}")
    
    # Train ensemble
    ensemble = HybridEnsemble(config)
    ensemble.fit(X, y, validation_split=0.2)
    
    # Evaluate
    metrics = ensemble.evaluate(X, y)
    
    # Feature importance
    imp_df = ensemble.get_feature_importance()
    if not imp_df.empty:
        logger.info("\nTop 10 Features:")
        logger.info(imp_df.head(10))
        imp_df.to_csv('data/processed/feature_importance.csv')
    
    # Save model
    os.makedirs('data/results', exist_ok=True)
    ensemble.save('data/results/ensemble_model.pkl')
    logger.info("✓ Model saved")
    
    return ensemble


def stage_backtesting(config: dict, ensemble, prices: pd.DataFrame,
                     features: pd.DataFrame, target: pd.Series):
    """Stage 4: Execute backtesting."""
    logger.info("\n" + "="*60)
    logger.info("STAGE 4: BACKTESTING & STRATEGY EXECUTION")
    logger.info("="*60)
    
    from src.backtest.backtest_engine import WalkForwardBacktest
    
    # Align data
    valid_idx = prices.index.intersection(features.index)
    prices_aligned = prices.loc[valid_idx]
    features_aligned = features.loc[valid_idx]
    target_aligned = target.loc[valid_idx]
    
    # Initialize backtest
    backtest = WalkForwardBacktest(config, ensemble, prices_aligned,
                                   features_aligned, target_aligned)
    
    # Run walk-forward validation
    wf_results = backtest.run()
    
    # Execute strategy
    strategy_results = backtest.execute_strategy()
    strategy_results.to_csv('data/results/strategy_results.csv', index=False)
    logger.info("✓ Strategy results saved")
    
    # Performance metrics
    metrics = backtest.compute_performance_metrics()
    
    # Save metrics
    import json
    with open('data/results/performance_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("✓ Performance metrics saved")
    
    return strategy_results, metrics


def main():
    """Main execution pipeline."""
    parser = argparse.ArgumentParser(
        description='Systematic Alpha Generation Framework'
    )
    parser.add_argument('--config', default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--stage', choices=[
        'data-fetch', 'feature-engineering', 'train-model',
        'backtest', 'analysis'
    ], help='Execute specific stage')
    parser.add_argument('--run-full', action='store_true',
                       help='Run complete pipeline')
    parser.add_argument('--walk-forward', action='store_true',
                       help='Enable walk-forward validation')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    logger.info("\n" + "="*60)
    logger.info("SYSTEMATIC ALPHA GENERATION")
    logger.info("="*60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Period: {config['data']['start_date']} to {config['data']['end_date']}")
    
    # Full pipeline
    if args.run_full:
        logger.info("Running FULL PIPELINE")
        
        # Stage 1: Data
        prices = stage_data_fetch(config)
        
        # Stage 2: Features
        features, target = stage_feature_engineering(config, prices)
        
        # Stage 3: Model
        ensemble = stage_model_training(config, features, target)
        
        # Stage 4: Backtest
        strategy_results, metrics = stage_backtesting(
            config, ensemble, prices, features, target
        )
        
        logger.info("\n" + "="*60)
        logger.info("✓ FULL PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"Results saved in: data/results/")
    
    # Individual stages
    elif args.stage == 'data-fetch':
        stage_data_fetch(config)
    
    elif args.stage == 'feature-engineering':
        prices = pd.read_csv('data/raw/prices_combined.csv', index_col=0, parse_dates=True)
        stage_feature_engineering(config, prices)
    
    elif args.stage == 'train-model':
        features = pd.read_csv('data/processed/features.csv', index_col=0, parse_dates=True)
        target = pd.read_csv('data/processed/target.csv', index_col=0, squeeze=True)
        stage_model_training(config, features, target)
    
    elif args.stage == 'backtest':
        from src.models.ensemble_model import HybridEnsemble
        
        prices = pd.read_csv('data/raw/prices_combined.csv', index_col=0, parse_dates=True)
        features = pd.read_csv('data/processed/features.csv', index_col=0, parse_dates=True)
        target = pd.read_csv('data/processed/target.csv', index_col=0, squeeze=True)
        ensemble = HybridEnsemble.load('data/results/ensemble_model.pkl')
        
        stage_backtesting(config, ensemble, prices, features, target)
    
    elif args.stage == 'analysis':
        logger.info("\nGenerating analysis reports...")
        # TODO: Implement analysis stage
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
