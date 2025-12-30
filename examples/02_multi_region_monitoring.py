#!/usr/bin/env python3
"""
Example 2: Multi-Region Monitoring
==================================

Monitor seismic risk across multiple regions simultaneously.

Author: Igor Chechelnitsky
License: CC BY-NC-ND 4.0 (Commercial Rights Reserved)
Contact: Facebook - Igor Chechelnitsky
"""

import sys
sys.path.insert(0, '..')

from src.qo3_system import QO3System, REGIONS
from datetime import datetime, timedelta
import json

def monitor_regions(regions=['japan', 'california', 'turkey', 'chile']):
    """
    Generate risk report for multiple regions.
    
    Parameters
    ----------
    regions : list
        List of region names to monitor
        
    Returns
    -------
    dict
        Consolidated risk reports
    """
    
    print("=" * 70)
    print("MULTI-REGION SEISMIC RISK MONITORING")
    print("=" * 70)
    
    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"\nAnalysis period: {start_date} to {end_date}")
    print(f"Regions: {', '.join(regions)}")
    
    reports = {}
    
    for region_name in regions:
        print(f"\n{'='*50}")
        print(f"Processing: {region_name.upper()}")
        print('='*50)
        
        try:
            # Initialize and run
            system = QO3System(region=region_name)
            system.fetch_data(start_date, end_date)
            system.compute_features()
            
            # Generate forecast
            forecast = system.generate_forecast(target_mag=5.5, horizon=14)
            reports[region_name] = forecast
            
            # Display summary
            print(f"  Events analyzed: {len(system.catalog)}")
            print(f"  Risk Level: {forecast['current_assessment']['risk_level']}")
            print(f"  QO3 Score: {forecast['current_assessment']['qo3_score']}")
            
            # Show active signals
            active_signals = [k for k, v in forecast['signals'].items() if v]
            if active_signals:
                print(f"  Active signals: {', '.join(active_signals)}")
            
        except Exception as e:
            print(f"  Error: {e}")
            reports[region_name] = {'error': str(e)}
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'Region':<15} {'Risk Level':<20} {'QO3 Score':<12} {'Signals'}")
    print("-" * 70)
    
    for region, report in reports.items():
        if 'error' not in report:
            level = report['current_assessment']['risk_level']
            score = report['current_assessment']['qo3_score']
            signals = sum(1 for v in report['signals'].values() if v)
            print(f"{region:<15} {level:<20} {score:<12.3f} {signals}/3")
        else:
            print(f"{region:<15} {'ERROR':<20} {'N/A':<12} N/A")
    
    # Save report
    output_file = 'multi_region_report.json'
    with open(output_file, 'w') as f:
        json.dump(reports, f, indent=2, default=str)
    
    print(f"\nReport saved to: {output_file}")
    
    return reports


if __name__ == "__main__":
    monitor_regions()
