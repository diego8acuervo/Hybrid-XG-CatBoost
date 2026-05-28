# Systematic Alpha Generation Using Machine Learning Ensembles

A production-ready implementation of the hybrid machine learning framework for predicting short-horizon market risk and generating systematic alpha, as described in "Causal and Predictive Modeling of Short-Horizon Market Risk and Systematic Alpha Generation Using Hybrid Machine Learning Ensembles" (Ranjan, 2025).

## Overview

This project implements a comprehensive trading framework that:

- **Predicts short-term SPY drawdowns** (5-day windows, >1% threshold) using cross-asset features
- **Generates systematic alpha** through a hybrid neural network + tree-based ensemble
- **Provides interpretability** via SHAP feature attribution analysis
- **Executes walk-forward backtesting** with proper temporal validation
- **Delivers risk-adjusted returns** (Sharpe: 2.51, Alpha: +0.28 annualized)

## Key Results (In-Sample 2005-2025)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 2.51 |
| Annualized Return | 40.84% |
| Annualized Volatility | 13.23% |
| Maximum Drawdown | -18.12% |
| CAPM Alpha (daily) | 0.00111 |
| CAPM Beta | 0.51 |
| Information Ratio | 1.73 |

## Architecture

```
systematic-alpha-generation/
├── config/                      # Configuration files
│   ├── config.yaml             # Main configuration
│   ├── feature_config.yaml     # Feature engineering specs
│   └── model_config.yaml       # Model hyperparameters
├── data/                       # Data storage
│   ├── raw/                    # Raw downloaded data
│   ├── processed/              # Processed features
│   └── results/                # Backtest results
├── src/                        # Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py         # Yahoo Finance data retrieval
│   │   ├── processor.py       # Feature engineering
│   │   └── validator.py       # Data validation
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technical.py       # Technical indicators
│   │   ├── statistical.py     # Moment-based features
│   │   ├── hurst.py           # Hurst exponent calculation
│   │   ├── cross_asset.py     # Cross-asset relationships
│   │   └── info_theory.py     # Information-theoretic measures
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py             # Neural network models
│   │   ├── tree_ensemble.py   # XGBoost, CatBoost
│   │   ├── voting.py          # Soft voting ensemble
│   │   └── training.py        # Training pipelines
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── shap_interpreter.py  # SHAP feature attribution
│   │   ├── performance.py       # Performance metrics
│   │   └── risk_metrics.py      # Risk calculations
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py            # Backtest engine
│   │   ├── strategy.py          # Strategy implementation
│   │   └── walk_forward.py      # Walk-forward validation
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Logging configuration
│       └── helpers.py           # Utility functions
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_backtest_analysis.ipynb
│   └── 05_shap_analysis.ipynb
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_backtest.py
├── Makefile                    # Build and task automation
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── main.py                     # Entry point
└── config.yaml                 # Default configuration
```

## Installation

### Prerequisites
- Python 3.9+
- pip or conda

### Setup

```bash
# Clone repository
git clone <repo-url>
cd systematic-alpha-generation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies using Makefile
make install

# Or manually
pip install -r requirements.txt

# Setup project (development mode)
make dev-setup
```

## Configuration

All configuration is centralized in `config/` directory:

### Main Configuration (`config/config.yaml`)
```yaml
# Data settings
data:
  start_date: "2005-01-01"
  end_date: "2025-01-13"
  symbols: ["SPY", "QQQ", "IWM", "TLT", "VIX", "GLD", "CL=F", 
            "DX-Y.NYB", "EURUSD=X", "JPYUSD=X", "TNX", "IRX"]
  cache_data: true
  cache_dir: "./data/raw"

# Feature engineering
features:
  rolling_windows: [21, 63]
  hurst_scales: [16, 64, 256]
  entropy_bins: 30
  target_drawdown: 0.01  # 1%
  target_horizon: 5      # 5 trading days
  feature_selection: "mutual_information"
  top_features: 80
  correlation_threshold: 0.95
  variance_threshold: 0.0001

# Model settings
model:
  ensemble_type: "soft_voting"
  base_learners: ["mlp", "xgboost", "catboost"]
  mlp_hidden_layers: [128, 64]
  mlp_dropout: 0.3
  xgboost_params:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.1
  catboost_params:
    iterations: 200
    depth: 6
    learning_rate: 0.1

# Backtest settings
backtest:
  strategy: "long_short"
  long_threshold: 0.5
  short_threshold: 0.5
  position_sizing: "probability_weighted"
  initial_capital: 100000
  transaction_cost: 0.001  # 0.1 bps

# Walk-forward testing
walk_forward:
  enabled: true
  train_size: 1000  # trading days
  test_size: 252    # 1 year
  rebalance_frequency: "weekly"
```

### Feature Configuration (`config/feature_config.yaml`)
Defines which features to engineer and their parameters.

### Model Configuration (`config/model_config.yaml`)
Hyperparameter tuning search spaces.

## Quick Start

### 1. Full Pipeline (End-to-End)
```bash
# Run complete pipeline: data download → feature engineering → model training → backtest
make run-full

# Or with Python
python main.py --config config/config.yaml --run-full
```

### 2. Data Download Only
```bash
python main.py --stage data-fetch
```

### 3. Feature Engineering
```bash
python main.py --stage feature-engineering
```

### 4. Model Training
```bash
python main.py --stage train-model
```

### 5. Walk-Forward Backtest
```bash
python main.py --stage backtest --walk-forward
```

### 6. Analysis & Reporting
```bash
python main.py --stage analysis --generate-report
```

## Usage Examples

### Python API

```python
from src.data.fetcher import YahooFinanceFetcher
from src.data.processor import FeatureProcessor
from src.models.voting import VotingEnsemble
from src.backtest.walk_forward import WalkForwardBacktest

# 1. Fetch data
fetcher = YahooFinanceFetcher(config)
data = fetcher.fetch_historical_data()

# 2. Engineer features
processor = FeatureProcessor(config)
features = processor.fit_transform(data)

# 3. Train ensemble model
ensemble = VotingEnsemble(config)
ensemble.fit(features, target)

# 4. Run walk-forward backtest
wf_backtest = WalkForwardBacktest(config, ensemble)
results = wf_backtest.run()
print(results.summary_statistics())

# 5. Generate SHAP explanations
from src.analysis.shap_interpreter import SHAPInterpreter
interpreter = SHAPInterpreter(ensemble)
interpreter.plot_summary(features)
```

### Notebook Workflow

```python
# notebooks/01_data_exploration.ipynb
# - Load configuration
# - Fetch data from Yahoo Finance
# - Visualize raw data and returns distributions
# - Summary statistics

# notebooks/02_feature_engineering.ipynb
# - Engineer all 178 features
# - Compute mutual information scores
# - Visualize top features
# - Feature selection and filtering

# notebooks/03_model_training.ipynb
# - Train MLP component
# - Train XGBoost and CatBoost
# - Hyperparameter optimization
# - Validation set performance

# notebooks/04_backtest_analysis.ipynb
# - Run walk-forward backtest
# - Plot cumulative returns
# - Equity curve and drawdown analysis
# - Performance attribution

# notebooks/05_shap_analysis.ipynb
# - SHAP values for crash weeks
# - SHAP values for non-crash weeks
# - Feature dependence plots
# - Decision boundary visualization
```

## Feature Engineering Details

### 1. Time-Series Moments
- Rolling volatility (21, 63-day windows)
- Skewness (return asymmetry)
- Kurtosis (tail heaviness)
- Shannon entropy (distribution uncertainty)

### 2. Hurst Exponent
- Multi-scale rescaled range analysis
- Captures long-term memory and mean-reversion
- Computed at scales: 16, 64, 256 trading days

### 3. Cross-Asset Relationships
- Rolling OLS beta with SPY
- Rolling Pearson correlation with SPY

### 4. Information-Theoretic Measures
- Kullback-Leibler divergence
- Measures distribution shifts vs. reference windows

### 5. Final Feature Space
- **178 engineered features** from 12 instruments
- **134 features** after low-variance/high-correlation filtering
- **80 features** selected via mutual information ranking

## Model Architecture

### Ensemble Components

1. **Multi-Layer Perceptron (MLP)**
   - 1-3 hidden layers (128-64 units)
   - ReLU activation
   - Dropout regularization (0.3)
   - Captures nonlinear temporal dependencies

2. **XGBoost Classifier**
   - 200 estimators, max_depth=6
   - Learning rate=0.1
   - Optimized for tabular data structure

3. **CatBoost Classifier**
   - 200 iterations, depth=6
   - Categorical feature handling
   - Reduced ordering bias

4. **Soft Voting Ensemble**
   - Averages predicted probabilities
   - Stabilizes predictions
   - Decision threshold: 0.5

## Backtesting Framework

### Strategy Logic

```
If P(crash) < 0.5:
    LONG position in SPY, scaled by confidence
Else If P(crash) >= 0.5:
    SHORT position in SPY, scaled by confidence
```

### Walk-Forward Validation

- **Train window**: 1000 trading days (~4 years)
- **Test window**: 252 trading days (1 year)
- **Rolling rebalance**: Weekly
- **No look-ahead bias**: Temporal splits preserved
- **Out-of-sample testing**: Rigorous forward testing protocol

### Performance Metrics Computed

```
Risk-Adjusted Returns:
- Sharpe Ratio
- Information Ratio (vs SPY)
- Sortino Ratio
- Calmar Ratio

Absolute Returns:
- Total Return
- Annualized Return
- Monthly/Daily Returns

Risk Metrics:
- Volatility (annualized)
- Maximum Drawdown
- Drawdown Duration
- Recovery Time

CAPM Metrics:
- Alpha (daily, annualized)
- Beta
- R-squared
- T-statistic (Alpha)

Trade Statistics:
- Win Rate
- Profit Factor
- Average Win/Loss
- Trade Duration
```

## Data Sources

### Primary Data: Yahoo Finance

All data retrieved via **yfinance** library:

| Asset Class | Symbols |
|------------|---------|
| Equities | SPY, QQQ, IWM, TLT |
| Volatility | VIX |
| Commodities | GLD, CL=F (WTI Crude) |
| FX | DX-Y.NYB, EURUSD=X, JPYUSD=X |
| Treasuries | TNX, IRX |

### Data Storage

- **Raw**: `data/raw/` (cached after first download)
- **Processed**: `data/processed/` (features, targets)
- **Results**: `data/results/` (backtest outputs, metrics)

### Loading Custom Data

The system supports CSV/Excel data input:

```python
# Load from CSV
processor.load_from_csv("data/raw/custom_prices.csv")

# Load from Excel
processor.load_from_excel("data/raw/custom_prices.xlsx")

# Specify column mappings
processor.set_column_mapping({
    "Date": "timestamp",
    "Adj Close": "price",
    "Volume": "volume"
})
```

## Development Guide

### Adding New Features

1. Create new module in `src/features/`
2. Implement feature computation class
3. Register in `FeatureProcessor.get_all_features()`
4. Add to `config/feature_config.yaml`

Example:
```python
# src/features/custom_indicator.py
class CustomIndicator:
    def __init__(self, **config):
        self.config = config
    
    def compute(self, data):
        # Your feature logic
        return feature_values
```

### Extending the Model

1. Create new model class in `src/models/`
2. Implement `fit()` and `predict()` methods
3. Add to `VotingEnsemble.base_learners`
4. Update configuration files

### Adding Tests

```bash
# Run all tests
make test

# Run specific test
pytest tests/test_features.py -v

# Coverage report
make test-coverage
```

## Troubleshooting

### Data Download Issues
```
Error: "No data for symbol XYZ"
Solution: Check Yahoo Finance availability, or provide local CSV
```

### Memory Issues with Large Feature Sets
```
Error: MemoryError during feature engineering
Solution: Reduce date range, or process in batches using batch_size parameter
```

### Model Training Fails
```
Error: "Insufficient samples for cross-validation"
Solution: Increase train_window in config, or reduce feature count
```

### Walk-Forward Results Inconsistent
```
Solution: Ensure random_state set in config for reproducibility
        Check temporal split logic in walk_forward.py
```

## Performance Considerations

### Optimization Tips

1. **Data Caching**: Enable in config to avoid repeated downloads
2. **Feature Subset**: Start with top 20 features for faster iteration
3. **Parallel Training**: Set `n_jobs=-1` in tree models
4. **Reduced Backtest**: Use smaller date range for testing
5. **GPU Acceleration**: CatBoost supports GPU (set `gpu_id` in config)

### Execution Time

| Stage | Time (Typical) |
|-------|---|
| Data Download | 2-5 min |
| Feature Engineering | 5-10 min |
| Model Training | 10-20 min |
| Walk-Forward Backtest | 30-60 min |
| SHAP Analysis | 5-15 min |
| Total | ~1-2 hours |

## Limitations & Caveats

### In-Sample Bias
Current results are **in-sample** (trained and tested on full 2005-2025 dataset). Walk-forward validation provides **out-of-sample** estimates.

### Transaction Costs
Performance metrics do NOT account for:
- Slippage (price impact)
- Bid-ask spreads
- Commissions (configurable in backtest.transaction_cost)

### Market Regime Changes
- Model trained on 2005-2025 (includes GFC, COVID, recent correction)
- Future market regimes may differ
- Periodic retraining recommended

### Survivorship Bias
- Using current index constituents
- Historical reconstitutions not accounted for

## Deployment

### Local Development

Run the FastAPI service locally for testing:

```bash
# Activate virtual environment
source venv/bin/activate

# Install API dependencies
pip install -r requirements.txt

# Start the API server
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

The API is now available at **http://localhost:8080** with interactive documentation at **http://localhost:8080/docs** (Swagger UI).

### API Endpoints

#### 1. Health Check
```bash
GET /health
```
Returns service status and model load timestamp. Returns 503 if model is not loaded.

#### 2. Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "features": [
    {
      "BTC-USD_vol_21d": 0.045,
      "BTC-USD_skew_21d": -0.15,
      ...
    }
  ],
  "threshold": 0.5
}
```
Response:
```json
{
  "predictions": [1],
  "probabilities": [0.37],
  "threshold": 0.5,
  "n_samples": 1,
  "timestamp": "2026-05-22T18:45:32.123456"
}
```

#### 3. Daily Batch Update
```bash
POST /batch-update
```
Triggers the full daily pipeline:
- Refreshes config with today's date as `end_date`
- Fetches latest prices from Yahoo Finance (all 12 symbols)
- Engineers all 178+ features
- Scores the most recent observation
- Returns crash probability + trading signal

Response:
```json
{
  "status": "ok",
  "target_symbol": "BTC-USD",
  "prediction_date": "2026-05-22",
  "crash_probability": 0.37,
  "signal": 1,
  "n_features_used": 80,
  "end_date_used": "2026-05-22"
}
```

**Note:** `signal = 1` means LOW crash risk (long position), `signal = -1` means HIGH crash risk (short position).

---

### Google Cloud Platform Deployment

Deploy to GCP using Docker, Cloud Run, and Vertex AI for monitoring.

#### Prerequisites
- GCP Project with billing enabled
- `gcloud` CLI installed and authenticated
- Docker installed locally

#### Step 1: Create a Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir google-cloud-storage google-cloud-bigquery

# Copy project files
COPY . .

# Expose port
EXPOSE 8080

# Start API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### Step 2: Setup GCP Infrastructure

```bash
# Set your project ID
PROJECT_ID="your-gcp-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    cloudscheduler.googleapis.com \
    aiplatform.googleapis.com \
    bigquery.googleapis.com
```

#### Step 3: Store Model Artifacts in Cloud Storage

```bash
# Create bucket
BUCKET_NAME="gs://${PROJECT_ID}-alpha-models"
gsutil mb $BUCKET_NAME

# Upload trained model
gsutil cp data/results/ensemble_model.pkl $BUCKET_NAME/
gsutil cp config/config.yaml $BUCKET_NAME/
```

Update `app.py` to download from GCS on startup:
```python
from google.cloud import storage

def _download_from_gcs(bucket_name, source_blob, dest_path):
    """Download file from GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob)
    blob.download_to_filename(dest_path)
```

#### Step 4: Build and Push Docker Image

```bash
# Create Artifact Registry repository
gcloud artifacts repositories create ml-models \
    --repository-format=docker \
    --location=us-central1

# Build image with Cloud Build
gcloud builds submit --tag us-central1-docker.pkg.dev/$PROJECT_ID/ml-models/alpha-model:v1

# (Alternative) Build locally and push
docker build -t alpha-model:v1 .
docker tag alpha-model:v1 us-central1-docker.pkg.dev/$PROJECT_ID/ml-models/alpha-model:v1
docker push us-central1-docker.pkg.dev/$PROJECT_ID/ml-models/alpha-model:v1
```

#### Step 5: Deploy to Cloud Run

```bash
gcloud run deploy alpha-model-service \
    --image us-central1-docker.pkg.dev/$PROJECT_ID/ml-models/alpha-model:v1 \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 600 \
    --set-env-vars "PORT=8080"

# Get the service URL
SERVICE_URL=$(gcloud run services describe alpha-model-service \
    --platform managed \
    --region us-central1 \
    --format 'value(status.url)')

echo "Service deployed at: $SERVICE_URL"
```

#### Step 6: Schedule Daily Batch Updates with Cloud Scheduler

```bash
# Create scheduler job (e.g., 1 AM UTC, weekdays only)
gcloud scheduler jobs create http daily-alpha-batch \
    --schedule="0 1 * * 1-5" \
    --uri="$SERVICE_URL/batch-update" \
    --http-method=POST \
    --time-zone="UTC" \
    --oidc-service-account-email=<service-account-email>
```

#### Step 7: Setup Vertex AI Model Monitoring

```bash
# Create BigQuery dataset for logging predictions
bq mk --dataset $PROJECT_ID:alpha_predictions

# Log predictions in app.py
from google.cloud import bigquery

def log_prediction(prediction_data):
    """Log prediction to BigQuery for monitoring."""
    client = bigquery.Client()
    table_id = f"{PROJECT_ID}.alpha_predictions.daily_scores"
    
    errors = client.insert_rows_json(table_id, [prediction_data])
    if errors:
        logger.error(f"BigQuery insert errors: {errors}")
```

Create Vertex AI monitoring job in GCP Console or via Python SDK:
```python
from google.cloud import aiplatform

# Setup monitoring that checks for training-serving skew
aiplatform.Model.register(
    display_name="alpha-model-monitor",
    artifact_uri=f"gs://{BUCKET_NAME}/ensemble_model.pkl",
    enable_monitoring=True,
)
```

---

### Production Checklist

- [ ] Model artifacts stored in Cloud Storage (versioned)
- [ ] Docker image built and pushed to Artifact Registry
- [ ] Cloud Run service deployed with appropriate memory/CPU
- [ ] Cloud Scheduler job configured for daily `/batch-update` calls
- [ ] BigQuery logging enabled for predictions
- [ ] Vertex AI monitoring configured (drift detection)
- [ ] Error alerting setup (Cloud Logging → Cloud Monitoring)
- [ ] Load testing performed (`locust` or similar)
- [ ] Secrets managed via Google Cloud Secret Manager
- [ ] Audit logging enabled for compliance

---

## Future Improvements

1. **Out-of-Sample Validation**
   - Implement proper forward-testing protocol
   - Walk-forward with retraining schedule

2. **Transaction Costs & Constraints**
   - Bid-ask spreads
   - Market impact models
   - Liquidity constraints
   - Margin requirements

3. **Alternative Data**
   - Options market indicators (put/call ratios, IV term structure)
   - Credit spreads (HY OAS)
   - Alternative assets (crypto, commodities futures)
   - Alternative data (sentiment, on-chain metrics)

4. **Advanced Architectures**
   - Transformer-based models
   - Graph neural networks for asset interactions
   - Attention mechanisms for feature importance

5. **Causal Analysis**
   - Granger causality testing
   - Causal forests
   - Instrumental variables for cross-asset relationships

6. **Robust Risk Management**
   - Stress testing scenarios
   - Historical simulation VaR/CVaR
   - Backtesting under market regimes (bull/bear/crisis)

7. **Deployment**
   - Real-time prediction serving (FastAPI)
   - Live signal generation
   - Alert system for high-risk periods
   - Integration with execution platforms



## References

Key papers and methodologies used:

1. Ranjan, A. (2025). "Causal and Predictive Modeling of Short-Horizon Market Risk and Systematic Alpha Generation Using Hybrid Machine Learning Ensembles"

2. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"

3. Dorogush, A. V., et al. (2018). "CatBoost: gradient boosting with categorical features support"

4. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions"

5. Cover, T. M., & Thomas, J. A. (2006). "Elements of Information Theory"

## License

MIT License - See LICENSE file

## Contact & Support

For questions, issues, or contributions:
- GitHub Issues: [Project Issues]
- Documentation: [Full Docs]
- Roadmap: [Development Roadmap]

## Disclaimer

This code is provided for **educational and research purposes only**. Past performance is not indicative of future results. Use at your own risk. Always conduct thorough backtesting and due diligence before deploying any trading strategy with real capital.
