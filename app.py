"""
FastAPI Web Service - Systematic Alpha Generation Model

Exposes the trained HybridEnsemble model via HTTP endpoints:
- GET  /            : Health check / service info
- GET  /health      : Liveness probe
- POST /predict     : Run inference on a feature payload
- POST /batch-update: Refresh data + features and run inference for today
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.data.fetcher import YahooFinanceFetcher
from src.features.feature_engineer import FeatureEngineer
from src.models.ensemble_model import HybridEnsemble

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("alpha-api")


# ---------------------------------------------------------------------------
# Paths & global state (populated on startup)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
MODEL_PATH = PROJECT_ROOT / "data" / "results" / "ensemble_model.pkl"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"

state: Dict = {
    "config": None,
    "ensemble": None,
    "feature_names": None,
    "loaded_at": None,
}


# ---------------------------------------------------------------------------
# Pydantic models for request / response
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """
    Request body for /predict.

    `features` is a dict mapping feature name -> value (single observation),
    or a list of such dicts (batch).
    """

    features: List[Dict[str, float]] = Field(
        ..., description="One or more feature vectors keyed by feature name."
    )
    threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Probability threshold for class 1."
    )


class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    threshold: float
    n_samples: int
    timestamp: str


class BatchUpdateResponse(BaseModel):
    status: str
    target_symbol: str
    prediction_date: str
    crash_probability: float
    signal: int
    n_features_used: int
    end_date_used: str


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    """Load YAML config and refresh end_date to today."""
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # Always use today's date as the most-recent end date
    today = datetime.utcnow().strftime("%Y-%m-%d")
    cfg.setdefault("data", {})["end_date"] = today
    logger.info(f"Config loaded. end_date set to {today}")
    return cfg


def _load_model() -> HybridEnsemble:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run `python main.py --run-full` first."
        )
    ensemble = HybridEnsemble.load(str(MODEL_PATH))
    logger.info(f"Model loaded from {MODEL_PATH}")
    return ensemble


def _initialize() -> None:
    """Load config + model into the global state at startup."""
    cfg = _load_config()
    ensemble = _load_model()

    state["config"] = cfg
    state["ensemble"] = ensemble
    state["feature_names"] = (
        list(ensemble.feature_names) if ensemble.feature_names is not None else None
    )
    state["loaded_at"] = datetime.utcnow().isoformat()
    logger.info(
        f"Initialization complete. Model expects "
        f"{len(state['feature_names']) if state['feature_names'] else 'unknown'} features."
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Systematic Alpha Generation API",
    description=(
        "HTTP interface for the Hybrid MLP + XGBoost + CatBoost ensemble that "
        "forecasts short-horizon crash probability for the configured target asset."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def on_startup() -> None:
    try:
        _initialize()
    except Exception as exc:
        # Keep the server up so /health can report the failure, but log loudly.
        logger.exception(f"Startup initialization failed: {exc}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Systematic Alpha Generation API",
        "version": "1.0.0",
        "model_loaded": state["ensemble"] is not None,
        "loaded_at": state["loaded_at"],
    }


@app.get("/health")
def health():
    if state["ensemble"] is None or state["config"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or config not loaded.",
        )
    return {"status": "ok", "loaded_at": state["loaded_at"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Run inference on one or more feature vectors.

    The caller must supply the same set of feature names the model was trained on.
    Missing features are filled with 0.0; extra features are ignored.
    """
    ensemble: Optional[HybridEnsemble] = state["ensemble"]
    expected_features: Optional[List[str]] = state["feature_names"]

    if ensemble is None or expected_features is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Check /health.",
        )

    if not req.features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`features` list must contain at least one observation.",
        )

    # Build a DataFrame aligned to the model's expected feature order.
    df = pd.DataFrame(req.features)
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0.0
    df = df[expected_features].astype(float).fillna(0.0)

    try:
        proba = ensemble.predict_proba(df)[:, 1]
        preds = (proba >= req.threshold).astype(int)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )

    return PredictResponse(
        predictions=preds.tolist(),
        probabilities=proba.tolist(),
        threshold=req.threshold,
        n_samples=len(df),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/batch-update", response_model=BatchUpdateResponse)
def batch_update():
    """
    Trigger the daily batch pipeline:
        1. Reload config with today's date as end_date.
        2. Fetch fresh prices via YahooFinanceFetcher.fetch_combined().
        3. Engineer features via FeatureEngineer.engineer_all_features().
        4. Score the most recent observation with the loaded ensemble.
        5. Return crash probability + trading signal for the target asset.
    """
    ensemble: Optional[HybridEnsemble] = state["ensemble"]
    expected_features: Optional[List[str]] = state["feature_names"]

    if ensemble is None or expected_features is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Check /health.",
        )

    # Refresh config so end_date == today
    cfg = _load_config()
    state["config"] = cfg
    target_symbol = cfg.get("data", {}).get("target_symbol", "BTC-USD")
    end_date_used = cfg["data"]["end_date"]

    try:
        # 1) Fetch latest prices
        logger.info("Batch update: fetching latest prices...")
        fetcher = YahooFinanceFetcher(cfg)
        prices = fetcher.fetch_combined()
        if prices.empty:
            raise RuntimeError("No price data returned by fetcher.")

        # 2) Engineer features
        logger.info("Batch update: engineering features...")
        engineer = FeatureEngineer(cfg)
        features = engineer.engineer_all_features(prices)

        # 3) Align with model's expected features
        for col in expected_features:
            if col not in features.columns:
                features[col] = 0.0
        features = features[expected_features].astype(float).fillna(0.0)

        if features.empty:
            raise RuntimeError("Feature matrix is empty after alignment.")

        # 4) Score the most recent row
        latest_row = features.iloc[[-1]]
        proba = float(ensemble.predict_proba(latest_row)[0, 1])

        # Strategy convention from backtest_engine: low crash prob -> LONG (1)
        signal = 1 if proba < 0.5 else -1

        prediction_date = str(latest_row.index[-1])
        logger.info(
            f"Batch update complete. {target_symbol} "
            f"P(crash)={proba:.4f} signal={signal} date={prediction_date}"
        )

        return BatchUpdateResponse(
            status="ok",
            target_symbol=target_symbol,
            prediction_date=prediction_date,
            crash_probability=proba,
            signal=signal,
            n_features_used=len(expected_features),
            end_date_used=end_date_used,
        )

    except Exception as exc:
        logger.exception("Batch update failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch update failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
