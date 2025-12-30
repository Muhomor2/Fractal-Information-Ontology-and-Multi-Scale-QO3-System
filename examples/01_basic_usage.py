#!/usr/bin/env python3
"""
Example 1: Basic QO3 Usage
==========================

Simple example demonstrating basic QO3 system usage for
seismic risk detection.

Author: Igor Chechelnitsky
License: CC BY-NC-ND 4.0 (Commercial Rights Reserved)
Contact: Facebook - Igor Chechelnitsky
"""

import sys
sys.path.insert(0, '..')

from src.qo3_system import QO3System

def main():
    """Basic QO3 usage example."""
    
    print("=" * 60)
    print("QO3 BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    # Initialize system for Japan
    print("\n1. Initializing QO3 system for Japan...")
    system = QO3System(region='japan')
    
    # Fetch data from USGS
    print("\n2. Fetching earthquake data from USGS...")
    system.fetch_data('2023-01-01', '2024-12-31')
    
    # Compute FIO features
    print("\n3. Computing FIO features...")
    features = system.compute_features()
    
    # Display feature summary
    print("\n4. Feature summary:")
    print(f"   Total days analyzed: {len(features)}")
    print(f"   b-value range: {features['b_value'].min():.2f} - {features['b_value'].max():.2f}")
    print(f"   CV range: {features['cv'].min():.2f} - {features['cv'].max():.2f}")
    
    # Generate forecast
    print("\n5. Generating current forecast...")
    forecast = system.generate_forecast(target_mag=5.0, horizon=7)
    
    # Display results
    print("\n" + "=" * 60)
    print("CURRENT RISK ASSESSMENT")
    print("=" * 60)
    print(f"\nRegion: {forecast['region']}")
    print(f"Date: {forecast['date']}")
    print(f"Target: M≥{forecast['target_magnitude']} within {forecast['horizon_days']} days")
    
    print(f"\n--- Risk Level ---")
    print(f"QO3 Score: {forecast['current_assessment']['qo3_score']}")
    print(f"Risk Level: {forecast['current_assessment']['risk_level']}")
    
    print(f"\n--- FIO Indicators ---")
    print(f"b-value: {forecast['current_assessment']['b_value']}")
    print(f"CV: {forecast['current_assessment']['cv']}")
    print(f"Entropy: {forecast['current_assessment']['entropy']}")
    
    print(f"\n--- Interpretation ---")
    print(f"{forecast['interpretation']}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
