#!/usr/bin/env python3
"""
QO3 Validation Module
=====================

Machine learning validation with strict temporal splitting,
proper metrics for rare events, and bootstrap confidence intervals.

Author: Igor Chechelnitsky
ORCID: 0009-0007-4607-1946
License: CC BY-NC-ND 4.0 (Commercial Rights Reserved)

Contact: Facebook - Igor Chechelnitsky
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

# ML imports (optional, graceful degradation)
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import (
        precision_recall_curve, roc_auc_score, average_precision_score,
        brier_score_loss, precision_score, recall_score, f1_score
    )
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. ML validation disabled.")


@dataclass
class ValidationResults:
    """Container for validation metrics."""
    baseline_rate: float
    precision_at_50_recall: float
    pr_auc: float
    roc_auc: float
    brier_score: float
    skill_score: float
    feature_importance: Dict[str, float]
    n_train: int
    n_test: int
    n_positive_test: int


class TemporalValidator:
    """
    Temporal cross-validation for seismic forecasting.
    
    Ensures strict time-ordering: all training data precedes test data.
    No information leakage from future to past.
    """
    
    def __init__(self, 
                 train_end_date: str,
                 features: List[str] = None):
        """
        Initialize validator.
        
        Parameters
        ----------
        train_end_date : str
            Last date for training (YYYY-MM-DD)
        features : list, optional
            Feature columns to use
        """
        self.train_end_date = pd.to_datetime(train_end_date).date()
        self.features = features or [
            'rate_7d', 'rate_14d', 'rate_30d',
            'b_value', 'cv', 'entropy',
            'b_value_change_7d', 'cv_change_7d',
            'rate_ratio_7_30', 'cv_distance_from_1'
        ]
        self.model = None
        self.scaler = None
        
    def prepare_data(self, 
                     df: pd.DataFrame,
                     target_col: str = 'target') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/test sets.
        
        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix with target
        target_col : str
            Target column name
            
        Returns
        -------
        tuple
            (train_df, test_df)
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        train = df[df['date'] <= self.train_end_date].copy()
        test = df[df['date'] > self.train_end_date].copy()
        
        return train, test
    
    def train(self, 
              train_df: pd.DataFrame,
              target_col: str = 'target') -> None:
        """
        Train the model on training data.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data
        target_col : str
            Target column name
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available")
        
        # Select available features
        available_features = [f for f in self.features if f in train_df.columns]
        
        X_train = train_df[available_features].copy()
        y_train = train_df[target_col].values
        
        # Handle missing values
        X_train = X_train.fillna(X_train.median())
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Gradient Boosting
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        self._feature_names = available_features
        
    def validate(self,
                 test_df: pd.DataFrame,
                 target_col: str = 'target') -> ValidationResults:
        """
        Validate model on test data.
        
        Parameters
        ----------
        test_df : pd.DataFrame
            Test data
        target_col : str
            Target column name
            
        Returns
        -------
        ValidationResults
            Comprehensive validation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        X_test = test_df[self._feature_names].copy()
        y_test = test_df[target_col].values
        
        # Handle missing values
        X_test = X_test.fillna(X_test.median())
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # === Metrics ===
        
        # Baseline rate
        baseline_rate = y_test.mean()
        
        # PR-AUC
        pr_auc = average_precision_score(y_test, y_prob)
        
        # ROC-AUC (only if both classes present)
        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = np.nan
        
        # Brier Score
        brier = brier_score_loss(y_test, y_prob)
        
        # Precision at 50% recall
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        idx_50_recall = np.argmin(np.abs(recall - 0.5))
        precision_at_50 = precision[idx_50_recall]
        
        # Skill Score = (precision - baseline) / (1 - baseline)
        if baseline_rate < 1:
            skill_score = (precision_at_50 - baseline_rate) / (1 - baseline_rate)
        else:
            skill_score = 0.0
        
        # Feature importance
        importance = dict(zip(self._feature_names, self.model.feature_importances_))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return ValidationResults(
            baseline_rate=baseline_rate,
            precision_at_50_recall=precision_at_50,
            pr_auc=pr_auc,
            roc_auc=roc_auc,
            brier_score=brier,
            skill_score=skill_score,
            feature_importance=importance,
            n_train=len(test_df) - len(test_df),
            n_test=len(test_df),
            n_positive_test=int(y_test.sum())
        )


def bootstrap_confidence_interval(y_true: np.ndarray,
                                   y_prob: np.ndarray,
                                   metric_func,
                                   n_bootstrap: int = 1000,
                                   ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Parameters
    ----------
    y_true : array
        True labels
    y_prob : array
        Predicted probabilities
    metric_func : callable
        Metric function(y_true, y_prob) -> float
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence interval level
        
    Returns
    -------
    tuple
        (point_estimate, lower_bound, upper_bound)
    """
    n = len(y_true)
    scores = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        try:
            score = metric_func(y_true[idx], y_prob[idx])
            scores.append(score)
        except:
            continue
    
    if len(scores) == 0:
        return np.nan, np.nan, np.nan
    
    point_estimate = np.mean(scores)
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    
    return point_estimate, lower, upper


def compute_fio_contribution(feature_importance: Dict[str, float]) -> float:
    """
    Compute percentage contribution of FIO components.
    
    FIO components: b_value, cv, entropy and their derivatives.
    Classical components: rate dynamics only.
    
    Parameters
    ----------
    feature_importance : dict
        Feature importance dictionary
        
    Returns
    -------
    float
        Percentage of importance from FIO components
    """
    fio_features = ['b_value', 'cv', 'entropy', 'b_value_change_7d', 
                    'cv_change_7d', 'entropy_change_7d', 'cv_distance_from_1', 'sid']
    
    total = sum(feature_importance.values())
    if total == 0:
        return 0.0
    
    fio_total = sum(v for k, v in feature_importance.items() 
                    if any(f in k for f in fio_features))
    
    return fio_total / total * 100


def print_validation_report(results: ValidationResults,
                            train_period: str,
                            test_period: str) -> None:
    """Print formatted validation report."""
    
    print("\n" + "=" * 60)
    print("QO3 VALIDATION REPORT")
    print("=" * 60)
    print(f"\nTraining period: {train_period}")
    print(f"Test period: {test_period}")
    print(f"Test samples: {results.n_test}")
    print(f"Positive events: {results.n_positive_test}")
    
    print("\n" + "-" * 40)
    print("PERFORMANCE METRICS")
    print("-" * 40)
    print(f"Baseline rate:         {results.baseline_rate:.1%}")
    print(f"Precision @50% recall: {results.precision_at_50_recall:.1%}")
    print(f"PR-AUC:                {results.pr_auc:.3f}")
    print(f"ROC-AUC:               {results.roc_auc:.3f}")
    print(f"Brier Score:           {results.brier_score:.3f}")
    print(f"Skill Score:           {results.skill_score:.3f}")
    
    print("\n" + "-" * 40)
    print("FEATURE IMPORTANCE")
    print("-" * 40)
    for feat, imp in list(results.feature_importance.items())[:10]:
        bar = "█" * int(imp * 50)
        print(f"  {feat:25s} {imp:5.1%} {bar}")
    
    fio_pct = compute_fio_contribution(results.feature_importance)
    print(f"\nFIO components total: {fio_pct:.1f}%")
    
    print("\n" + "=" * 60)
