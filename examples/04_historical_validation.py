#!/usr/bin/env python3
"""
Example 4: Historical Validation (Backtesting)
==============================================

Validate QO3 performance on historical data with proper
train/test temporal split and statistical metrics.

Author: Igor Chechelnitsky
License: CC BY-NC-ND 4.0 (Commercial Rights Reserved)
Contact: Facebook - Igor Chechelnitsky
"""

import sys
sys.path.insert(0, '..')

from src.qo3_system import QO3System, create_target_variable
from src.validation import (
    TemporalValidator, 
    print_validation_report,
    bootstrap_confidence_interval,
    compute_fio_contribution
)
from datetime import datetime, timedelta

def run_validation():
    """Run comprehensive validation on historical data."""
    
    print("=" * 70)
    print("QO3 HISTORICAL VALIDATION (BACKTESTING)")
    print("=" * 70)
    
    # Setup parameters
    region = 'tohoku'
    target_mag = 5.0
    horizon = 7
    train_end = '2022-06-30'
    
    print(f"\nConfiguration:")
    print(f"  Region: {region}")
    print(f"  Target magnitude: M≥{target_mag}")
    print(f"  Forecast horizon: {horizon} days")
    print(f"  Train/test split: {train_end}")
    
    # Initialize and fetch data
    print("\n1. Fetching historical data...")
    system = QO3System(region=region)
    system.fetch_data('2017-01-01', '2024-12-31')
    
    print(f"   Total events: {len(system.catalog)}")
    
    # Compute features
    print("\n2. Computing FIO features...")
    features = system.compute_features()
    print(f"   Total days: {len(features)}")
    
    # Create target variable (strictly future-looking)
    print("\n3. Creating target variable...")
    features_with_target = create_target_variable(
        system.catalog, 
        features, 
        target_mag=target_mag, 
        horizon=horizon
    )
    
    positive_rate = features_with_target['target'].mean()
    print(f"   Positive rate (baseline): {positive_rate:.1%}")
    print(f"   Positive samples: {features_with_target['target'].sum()}")
    
    # Temporal split
    print("\n4. Preparing train/test split...")
    validator = TemporalValidator(train_end_date=train_end)
    train_df, test_df = validator.prepare_data(features_with_target)
    
    print(f"   Training samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    print(f"   Test positives: {test_df['target'].sum()}")
    
    # Check if enough data
    if len(train_df) < 100:
        print("\n   WARNING: Insufficient training data!")
        return
    if len(test_df) < 30:
        print("\n   WARNING: Insufficient test data!")
        return
    
    # Train model
    print("\n5. Training Gradient Boosting model...")
    validator.train(train_df)
    
    # Validate
    print("\n6. Validating on test set...")
    results = validator.validate(test_df)
    
    # Print detailed report
    print_validation_report(
        results,
        train_period="2017-01 to 2022-06",
        test_period="2022-07 to 2024-12"
    )
    
    # Additional analysis
    print("\n" + "=" * 70)
    print("ADDITIONAL ANALYSIS")
    print("=" * 70)
    
    # FIO contribution
    fio_pct = compute_fio_contribution(results.feature_importance)
    print(f"\nFIO component contribution: {fio_pct:.1f}%")
    print("(FIO = b-value, CV, entropy and their derivatives)")
    
    # Improvement over baseline
    improvement = (results.pr_auc / results.baseline_rate - 1) * 100
    print(f"\nPR-AUC improvement over baseline: {improvement:.1f}%")
    
    # Interpretation
    print("\n" + "-" * 40)
    print("INTERPRETATION")
    print("-" * 40)
    
    if results.skill_score > 0.05:
        print("✓ Significant predictive skill detected")
    else:
        print("○ Marginal predictive skill")
    
    if fio_pct > 40:
        print("✓ FIO features provide substantial contribution")
    else:
        print("○ Rate dynamics dominate over FIO features")
    
    if results.roc_auc > 0.6:
        print("✓ Good discrimination between risk levels")
    else:
        print("○ Limited discrimination ability")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_validation()
