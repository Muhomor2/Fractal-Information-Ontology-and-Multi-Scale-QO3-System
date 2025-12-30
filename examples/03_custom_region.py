#!/usr/bin/env python3
"""
Example 3: Custom Region Analysis
=================================

Analyze a custom region not in the predefined list.

Author: Igor Chechelnitsky
License: CC BY-NC-ND 4.0 (Commercial Rights Reserved)
Contact: Facebook - Igor Chechelnitsky
"""

import sys
sys.path.insert(0, '..')

from src.qo3_system import QO3System, RegionConfig

def analyze_custom_region():
    """Demonstrate custom region definition and analysis."""
    
    print("=" * 60)
    print("CUSTOM REGION ANALYSIS")
    print("=" * 60)
    
    # Define custom region: Eastern Mediterranean
    print("\n1. Defining custom region: Eastern Mediterranean...")
    
    custom_region = RegionConfig(
        name="Eastern Mediterranean",
        lat_min=32.0,
        lat_max=42.0,
        lon_min=20.0,
        lon_max=40.0,
        mc=3.5,  # Magnitude of completeness
        description="Eastern Mediterranean including Greece, Turkey, Cyprus"
    )
    
    print(f"   Name: {custom_region.name}")
    print(f"   Bounds: {custom_region.lat_min}°-{custom_region.lat_max}°N, "
          f"{custom_region.lon_min}°-{custom_region.lon_max}°E")
    print(f"   Mc: {custom_region.mc}")
    
    # Initialize with custom region
    print("\n2. Initializing QO3 with custom region...")
    system = QO3System(region=custom_region)
    
    # Fetch and analyze
    print("\n3. Fetching and analyzing data...")
    system.fetch_data('2022-01-01', '2025-01-01')
    system.compute_features()
    
    print(f"   Events retrieved: {len(system.catalog)}")
    print(f"   Days analyzed: {len(system.features)}")
    
    # Generate forecast
    print("\n4. Generating forecast...")
    forecast = system.generate_forecast(target_mag=5.0, horizon=7)
    
    # Display results
    print("\n" + "=" * 60)
    print(f"RISK ASSESSMENT: {custom_region.name}")
    print("=" * 60)
    
    print(f"\nDate: {forecast['date']}")
    print(f"Target: M≥{forecast['target_magnitude']} within {forecast['horizon_days']} days")
    
    print(f"\n--- Current Assessment ---")
    print(f"Risk Level: {forecast['current_assessment']['risk_level']}")
    print(f"QO3 Score: {forecast['current_assessment']['qo3_score']}")
    print(f"b-value: {forecast['current_assessment']['b_value']}")
    print(f"CV: {forecast['current_assessment']['cv']}")
    
    print(f"\n--- Interpretation ---")
    print(f"{forecast['interpretation']}")
    
    # Another example: Dead Sea Transform (Israel region)
    print("\n" + "=" * 60)
    print("SECOND CUSTOM REGION: Dead Sea Transform")
    print("=" * 60)
    
    dst_region = RegionConfig(
        name="Dead Sea Transform",
        lat_min=29.0,
        lat_max=34.0,
        lon_min=34.0,
        lon_max=37.0,
        mc=2.5,
        description="Dead Sea Transform fault system"
    )
    
    system2 = QO3System(region=dst_region)
    system2.fetch_data('2020-01-01', '2025-01-01')
    system2.compute_features()
    
    forecast2 = system2.generate_forecast(target_mag=4.0, horizon=14)
    
    print(f"\nRegion: {dst_region.name}")
    print(f"Events: {len(system2.catalog)}")
    print(f"Risk Level: {forecast2['current_assessment']['risk_level']}")
    print(f"QO3 Score: {forecast2['current_assessment']['qo3_score']}")


if __name__ == "__main__":
    analyze_custom_region()
