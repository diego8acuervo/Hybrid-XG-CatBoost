"""
Hybrid Ensemble Model Implementation

Combines MLP neural network with XGBoost and CatBoost
for robust classification of crash vs non-crash regimes.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, List
import warnings

# ML Libraries
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

import joblib

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class HybridEnsemble:
    """
    Hybrid ensemble combining neural networks and tree-based models.
    
    Architecture:
    - MLP: Captures temporal non-linearities
    - XGBoost: Robust tree-based learning
    - CatBoost: Categorical handling and interpretability
    - Soft Voting: Averaged probability predictions
    """
    
    def __init__(self, config: dict):
        """Initialize ensemble."""
        self.config = config.get('model', {})
        self.ensemble_type = self.config.get('ensemble_type', 'soft_voting')
        self.random_state = self.config.get('random_state', 42)
        
        # Initialize base learners
        self.mlp = None
        self.xgboost = None
        self.catboost = None
        self.scaler = StandardScaler()
        
        self.trained_models = []
        self.feature_names = None
    
    def build_mlp(self) -> MLPClassifier:
        """Build MLP neural network."""
        hidden_layers = self.config.get('mlp_hidden_layers', [128, 64])
        dropout = self.config.get('mlp_dropout', 0.3)
        
        mlp = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layers),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            batch_size=32,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=self.random_state,
            verbose=0
        )
        
        logger.info(f"MLP built: {hidden_layers}")
        return mlp
    
    def build_xgboost(self) -> XGBClassifier:
        """Build XGBoost classifier."""
        if XGBClassifier is None:
            logger.warning("XGBoost not installed, skipping")
            return None
        
        xgb_params = self.config.get('xgboost_params', {})
        
        xgb = XGBClassifier(
            n_estimators=xgb_params.get('n_estimators', 200),
            max_depth=xgb_params.get('max_depth', 6),
            learning_rate=xgb_params.get('learning_rate', 0.1),
            subsample=xgb_params.get('subsample', 0.8),
            colsample_bytree=xgb_params.get('colsample_bytree', 0.8),
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=self.random_state,
            verbosity=0,
            n_jobs=-1
        )
        
        logger.info("XGBoost built")
        return xgb
    
    def build_catboost(self) -> CatBoostClassifier:
        """Build CatBoost classifier."""
        if CatBoostClassifier is None:
            logger.warning("CatBoost not installed, skipping")
            return None
        
        cat_params = self.config.get('catboost_params', {})
        
        cat = CatBoostClassifier(
            iterations=cat_params.get('iterations', 200),
            depth=cat_params.get('depth', 6),
            learning_rate=cat_params.get('learning_rate', 0.1),
            loss_function='Logloss',
            verbose=False,
            random_state=self.random_state,
            thread_count=-1
        )
        
        logger.info("CatBoost built")
        return cat
    
    def fit(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> None:
        """
        Train ensemble on data with time-series cross-validation.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            validation_split: Portion for validation
        """
        logger.info(f"Training ensemble on {X.shape[0]} samples, {X.shape[1]} features")
        
        self.feature_names = X.columns
        
        # Time series split
        n_train = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:n_train], X.iloc[n_train:]
        y_train, y_val = y.iloc[:n_train], y.iloc[n_train:]
        
        # Standardize
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
        logger.info(f"Positive class: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
        
        # Train MLP
        logger.info("Training MLP...")
        self.mlp = self.build_mlp()
        self.mlp.fit(X_train_scaled, y_train)
        mlp_auc = roc_auc_score(y_val, self.mlp.predict_proba(X_val_scaled)[:, 1])
        logger.info(f"  MLP ROC-AUC (val): {mlp_auc:.4f}")
        self.trained_models.append(('MLP', self.mlp))
        
        # Train XGBoost
        logger.info("Training XGBoost...")
        self.xgboost = self.build_xgboost()
        if self.xgboost is not None:
            self.xgboost.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            xgb_auc = roc_auc_score(y_val, self.xgboost.predict_proba(X_val)[:, 1])
            logger.info(f"  XGBoost ROC-AUC (val): {xgb_auc:.4f}")
            self.trained_models.append(('XGBoost', self.xgboost))
        
        # Train CatBoost
        logger.info("Training CatBoost...")
        self.catboost = self.build_catboost()
        if self.catboost is not None:
            self.catboost.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False
            )
            cat_auc = roc_auc_score(y_val, self.catboost.predict_proba(X_val)[:, 1])
            logger.info(f"  CatBoost ROC-AUC (val): {cat_auc:.4f}")
            self.trained_models.append(('CatBoost', self.catboost))
        
        logger.info("✓ Ensemble training complete")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict crash probability via soft voting.
        
        Returns: Array of probabilities [n_samples, 2]
        """
        X_scaled = self.scaler.transform(X)
        
        probas = []
        
        # MLP prediction
        if self.mlp is not None:
            mlp_proba = self.mlp.predict_proba(X_scaled)[:, 1]
            probas.append(mlp_proba)
        
        # XGBoost prediction
        if self.xgboost is not None:
            xgb_proba = self.xgboost.predict_proba(X)[:, 1]
            probas.append(xgb_proba)
        
        # CatBoost prediction
        if self.catboost is not None:
            cat_proba = self.catboost.predict_proba(X)[:, 1]
            probas.append(cat_proba)
        
        # Soft voting: average probabilities
        ensemble_proba = np.mean(probas, axis=0)
        
        return np.column_stack([1 - ensemble_proba, ensemble_proba])
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate ensemble performance."""
        y_pred_proba = self.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'ROC-AUC': roc_auc_score(y, y_pred_proba),
            'Precision': precision_score(y, y_pred),
            'Recall': recall_score(y, y_pred),
            'F1': f1_score(y, y_pred),
            'Confusion Matrix': confusion_matrix(y, y_pred)
        }
        
        logger.info("=" * 50)
        logger.info("ENSEMBLE PERFORMANCE")
        logger.info("=" * 50)
        logger.info(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")
        logger.info(f"Precision: {metrics['Precision']:.4f}")
        logger.info(f"Recall: {metrics['Recall']:.4f}")
        logger.info(f"F1-Score: {metrics['F1']:.4f}")
        logger.info(classification_report(y, y_pred))
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance from tree models."""
        importances = {}
        
        if self.xgboost is not None:
            xgb_imp = self.xgboost.feature_importances_
            importances['XGBoost'] = xgb_imp
        
        if self.catboost is not None:
            cat_imp = self.catboost.feature_importances_
            importances['CatBoost'] = cat_imp
        
        if not importances:
            logger.warning("No tree models available for feature importance")
            return pd.DataFrame()
        
        imp_df = pd.DataFrame(importances, index=self.feature_names)
        imp_df['Mean'] = imp_df.mean(axis=1)
        imp_df = imp_df.sort_values('Mean', ascending=False)
        
        return imp_df
    
    def save(self, filepath: str) -> None:
        """Save trained ensemble."""
        joblib.dump(self, filepath)
        logger.info(f"Ensemble saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'HybridEnsemble':
        """Load trained ensemble."""
        ensemble = joblib.load(filepath)
        logger.info(f"Ensemble loaded from {filepath}")
        return ensemble
