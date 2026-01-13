# Implementation Guide: Systematic Alpha Generation

## Quick Start (5 minutes)

```bash
# 1. Clone and navigate
cd systematic-alpha-generation

# 2. Setup environment
make install

# 3. Run full pipeline
make run-full

# Results saved to: data/results/
```

## Project Structure

The code is organized following Python best practices:

```
src/
├── data/
│   ├── fetcher.py         # Yahoo Finance integration
│   ├── processor.py       # Data preprocessing
│   └── validator.py       # Data quality checks
├── features/
│   └── feature_engineer.py  # All 178+ feature engineering
├── models/
│   ├── ensemble_model.py  # Hybrid MLP + XGBoost + CatBoost
│   └── training.py        # Training pipelines
├── backtest/
│   ├── backtest_engine.py # Walk-forward validation
│   └── strategy.py        # Trading strategy logic
└── analysis/
    ├── shap_interpreter.py  # Feature attribution
    └── performance.py       # Metrics computation
```

## Core Components

### 1. Data Fetching (`data/fetcher.py`)

**YahooFinanceFetcher**: Downloads OHLCV data for 12 symbols across 5 asset classes

```python
from src.data.fetcher import YahooFinanceFetcher
config = load_config('config/config.yaml')
fetcher = YahooFinanceFetcher(config)
prices = fetcher.fetch_combined()  # Returns DataFrame
```

**Features:**
- Caching to avoid redundant API calls
- Batch downloading with progress tracking
- Automatic data validation
- CSV export for local data loading

### 2. Feature Engineering (`src/features/feature_engineer.py`)

**FeatureEngineer**: Implements all 178+ features from the paper

```python
from src.features.feature_engineer import FeatureEngineer
engineer = FeatureEngineer(config)
features = engineer.engineer_all_features(prices)
target = engineer.create_target_variable(prices)
```

**Feature Categories:**

1. **Time-Series Moments** (Statistical Features)
   - Volatility (std of returns)
   - Skewness (3rd moment)
   - Kurtosis (excess 4th moment)
   - Shannon Entropy
   - Rolling windows: 21, 63 days

2. **Hurst Exponent** (Persistence/Mean-Reversion)
   - Multi-scale rescaled range analysis
   - Scales: 16, 64, 256 trading days
   - H > 0.5: Trending, H < 0.5: Mean-reverting

3. **Cross-Asset Relationships**
   - Rolling beta vs SPY
   - Rolling correlation vs SPY
   - For all non-SPY assets

4. **Information-Theoretic Measures**
   - KL divergence vs reference distribution
   - Captures distribution shifts
   - Window pairs: (21, 126), (63, 126)

**Feature Selection:**
```python
# Filter low-variance features
features = engineer.select_features_by_variance(features, threshold=1e-4)

# Remove highly correlated pairs
features = engineer.select_features_by_correlation(features, threshold=0.95)

# Select top 80 by mutual information
features, mi_scores = engineer.select_top_features_by_mi(
    features, target, n_features=80
)

# Standardize (mean=0, std=1)
features = engineer.standardize_features(features, fit=True)
```

### 3. Ensemble Model (`src/models/ensemble_model.py`)

**HybridEnsemble**: Combines 3 model types via soft voting

```python
from src.models.ensemble_model import HybridEnsemble

ensemble = HybridEnsemble(config)
ensemble.fit(X_train, y_train, validation_split=0.2)
predictions = ensemble.predict_proba(X_test)
```

**Base Learners:**

1. **MLP Neural Network**
   - Hidden layers: [128, 64]
   - Activation: ReLU
   - Dropout: 0.3
   - Captures temporal non-linearities

2. **XGBoost Classifier**
   - 200 estimators, max_depth=6
   - Learning rate: 0.1
   - Robust tree-based learning

3. **CatBoost Classifier**
   - 200 iterations, depth=6
   - Categorical feature handling
   - Reduced ordering bias

**Ensemble Strategy:**
```
P(crash) = mean([MLP_prob, XGBoost_prob, CatBoost_prob])
Prediction = 1 if P(crash) > 0.5 else 0
```

### 4. Walk-Forward Backtesting (`src/backtest/backtest_engine.py`)

**WalkForwardBacktest**: Proper temporal validation avoiding look-ahead bias

```python
from src.backtest.backtest_engine import WalkForwardBacktest

backtest = WalkForwardBacktest(config, ensemble, prices, features, target)

# Run walk-forward splits
wf_results = backtest.run()

# Execute strategy
strategy_df = backtest.execute_strategy()

# Compute metrics
metrics = backtest.compute_performance_metrics()
```

**Walk-Forward Logic:**

```
For each split:
  1. Train on historical data (1000 trading days ~4 years)
  2. Test on forward period (252 trading days ~1 year)
  3. Rolling window: Move forward by test_size
  4. No look-ahead bias
  5. Track out-of-sample performance

Example splits:
  Split 1: Train[0:1000], Test[1000:1252]
  Split 2: Train[252:1252], Test[1252:1504]
  ...
```

**Trading Strategy:**

```
IF P(crash in next 5 days) < 0.5:
    LONG position, sized by confidence
ELSE IF P(crash) >= 0.5:
    SHORT position, sized by confidence

Position Size = |P(crash) - 0.5| * 2  (range [0, 1])
```

**Performance Metrics:**

```
Risk-Adjusted Returns:
- Sharpe Ratio (annualized)
- Information Ratio (vs SPY)
- Sortino Ratio
- Calmar Ratio

Absolute Returns:
- Total Return (%)
- Annualized Return (%)

Risk Metrics:
- Annual Volatility
- Maximum Drawdown
- Win Rate

CAPM Metrics:
- Alpha (annualized + daily)
- Beta (market exposure)
- R-squared
- T-statistic (significance)
```

## Configuration

Main configuration in `config/config.yaml`:

### Data Settings
```yaml
data:
  start_date: "2005-01-01"
  end_date: "2025-01-13"
  symbols: [SPY, QQQ, IWM, ..., IRX]
  cache_data: true
```

### Feature Settings
```yaml
features:
  rolling_windows: [21, 63]
  hurst_scales: [16, 64, 256]
  top_features: 80
  correlation_threshold: 0.95
```

### Model Settings
```yaml
model:
  ensemble_type: "soft_voting"
  mlp_hidden_layers: [128, 64]
  xgboost_params:
    n_estimators: 200
    max_depth: 6
```

### Backtest Settings
```yaml
backtest:
  strategy: "long_short"
  long_threshold: 0.5
  position_sizing: "probability_weighted"
```

### Walk-Forward Settings
```yaml
walk_forward:
  train_size: 1000  # ~4 years
  test_size: 252    # ~1 year
  rebalance_frequency: "weekly"
```

## Execution Pipeline

### Full Execution
```bash
python main.py --config config/config.yaml --run-full --verbose
```

### Stage by Stage

**Stage 1: Data Fetching**
```bash
python main.py --stage data-fetch
# Downloads data for all symbols (2005-2025)
# Saves to: data/raw/prices_combined.csv
# Time: ~2-5 minutes
```

**Stage 2: Feature Engineering**
```bash
python main.py --stage feature-engineering
# Engineers 178+ features
# Applies MI feature selection (top 80)
# Saves to: data/processed/features.csv
# Time: ~5-10 minutes
```

**Stage 3: Model Training**
```bash
python main.py --stage train-model
# Trains MLP + XGBoost + CatBoost ensemble
# Validates on holdout set (20%)
# Saves to: data/results/ensemble_model.pkl
# Time: ~10-20 minutes
```

**Stage 4: Backtesting**
```bash
python main.py --stage backtest --walk-forward
# Runs walk-forward validation
# Executes strategy
# Computes performance metrics
# Saves to: data/results/
# Time: ~30-60 minutes
```

## Output Files

After running the pipeline:

```
data/
├── raw/
│   └── prices_combined.csv        # Downloaded OHLCV data
├── processed/
│   ├── features.csv               # Engineered feature matrix
│   ├── target.csv                 # Binary target (crash indicator)
│   └── feature_importance.csv     # Feature importance ranking
└── results/
    ├── ensemble_model.pkl         # Trained model
    ├── strategy_results.csv       # Daily P&L, signals, positions
    ├── performance_metrics.json   # Performance statistics
    └── backtest_report.html       # Interactive report (optional)
```

## Key Results (In-Sample 2005-2025)

Based on the paper's methodology:

```
Sharpe Ratio:           2.51
Annualized Return:      40.84%
Annualized Volatility:  13.23%
Maximum Drawdown:       -18.12%
Information Ratio:      1.73
CAPM Beta:              0.51
Alpha (Annualized):     +0.28 (28 bps)
```

## Advanced Usage

### Using SHAP for Explainability

```python
import shap
from src.analysis.shap_interpreter import SHAPInterpreter

interpreter = SHAPInterpreter(ensemble)

# Feature importance
interpreter.plot_summary(features)

# Force plots
interpreter.plot_force(features.iloc[0])

# Dependence plots
interpreter.plot_dependence('CL=F_hurst_short', features)
```

### Custom Data Loading

```python
from src.data.fetcher import CSVDataLoader

loader = CSVDataLoader('path/to/data/')
data = loader.load_all(symbols)
```

### Running Multiple Experiments

```python
import itertools
import json

# Parameter grid
param_grid = {
    'top_features': [50, 80, 120],
    'mlp_hidden_layers': [[128, 64], [256, 128]],
    'xgboost_params': [...]
}

results = {}

for params in itertools.product(*param_grid.values()):
    config_exp = config.copy()
    # Update config with params
    
    # Run pipeline
    metrics = run_pipeline(config_exp)
    results[str(params)] = metrics

# Save results
with open('experiment_results.json', 'w') as f:
    json.dump(results, f)
```

## Troubleshooting

### Issue: Memory Error During Feature Engineering

**Solution:** Process in batches or reduce date range:
```python
# Process last 5 years only
config['data']['start_date'] = '2020-01-01'
```

### Issue: No Data for Symbol

**Solution:** Check Yahoo Finance availability or provide local CSV:
```python
loader = CSVDataLoader('./data/raw/')
data = loader.load_all(symbols)
```

### Issue: Model Training Fails

**Solution:** Ensure sufficient data:
```python
# Minimum required samples
min_samples = walk_forward['train_size'] + walk_forward['test_size']
# Should be < len(features)
```

### Issue: Walk-Forward Takes Too Long

**Solution:** Reduce number of splits or use smaller date range:
```yaml
walk_forward:
  train_size: 500    # Smaller training window
  test_size: 126     # Smaller test window
```

## Performance Optimization

1. **Enable Data Caching**
   ```yaml
   data:
     cache_data: true
   ```

2. **Use Feature Subset for Testing**
   ```yaml
   features:
     top_features: 20  # Start with top 20
   ```

3. **Reduce Backtest Period**
   ```python
   config['data']['end_date'] = '2023-01-01'  # 2 years instead of 20
   ```

4. **Parallel Processing**
   ```python
   # Set n_jobs=-1 in XGBoost/CatBoost
   xgboost_params:
     n_jobs: -1
   ```

## Next Steps & Improvements

1. **Out-of-Sample Validation**
   - Implement live/paper trading
   - Track real-world performance
   - Adapt to market regime changes

2. **Risk Management**
   - Incorporate position limits
   - Dynamic stop-losses
   - Portfolio-level constraints

3. **Advanced Features**
   - Options implied volatility
   - Credit spreads
   - Alternative data (sentiment, on-chain)

4. **Deployment**
   - REST API for predictions
   - Real-time signal generation
   - Integration with trading platforms

5. **Causal Analysis**
   - Granger causality testing
   - Causal forests
   - Structural break detection

## References

### Papers
- Ranjan, A. (2025). "Causal and Predictive Modeling of Short-Horizon Market Risk"
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
- Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions" (SHAP)

### Libraries
- [scikit-learn](https://scikit-learn.org/) - ML pipelines
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting
- [CatBoost](https://catboost.ai/) - Categorical boosting
- [SHAP](https://shap.readthedocs.io/) - Model interpretability
- [yfinance](https://github.com/ranaryan/yfinance) - Yahoo Finance API

## Support

For issues or questions:
1. Check logs in `alpha_generation.log`
2. Review configuration in `config/config.yaml`
3. Run with `--verbose` flag for detailed output

---

**Last Updated:** January 2025
**Version:** 1.0.0
