#!/usr/bin/env python3
"""
Example 5: Automated Alert System
=================================

Simple alert system that monitors risk levels and generates
notifications for elevated risk regions.

Author: Igor Chechelnitsky
License: CC BY-NC-ND 4.0 (Commercial Rights Reserved)
Contact: Facebook - Igor Chechelnitsky
"""

import sys
sys.path.insert(0, '..')

from src.qo3_system import QO3System
from datetime import datetime, timedelta
import json

class QO3AlertSystem:
    """
    Automated seismic risk alert system.
    
    Monitors configured regions and generates alerts
    when QO3 score exceeds threshold.
    """
    
    def __init__(self, 
                 regions=['japan', 'california'],
                 target_mag=5.5,
                 horizon=7,
                 alert_threshold=0.5):
        """
        Initialize alert system.
        
        Parameters
        ----------
        regions : list
            Regions to monitor
        target_mag : float
            Target magnitude for forecast
        horizon : int
            Forecast horizon in days
        alert_threshold : float
            QO3 score threshold for alerts (0-1)
        """
        self.regions = regions
        self.target_mag = target_mag
        self.horizon = horizon
        self.alert_threshold = alert_threshold
        
    def check_all_regions(self):
        """
        Check all configured regions for elevated risk.
        
        Returns
        -------
        list
            List of alert dictionaries for high-risk regions
        """
        alerts = []
        
        # Date range (6 months of data)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        for region in self.regions:
            try:
                # Initialize and analyze
                system = QO3System(region=region)
                system.fetch_data(start_date, end_date)
                system.compute_features()
                
                # Generate forecast
                forecast = system.generate_forecast(
                    target_mag=self.target_mag, 
                    horizon=self.horizon
                )
                
                score = forecast['current_assessment']['qo3_score']
                
                # Check threshold
                if score is not None and score >= self.alert_threshold:
                    alerts.append({
                        'region': region,
                        'date': forecast['date'],
                        'qo3_score': score,
                        'risk_level': forecast['current_assessment']['risk_level'],
                        'b_value': forecast['current_assessment']['b_value'],
                        'cv': forecast['current_assessment']['cv'],
                        'signals': forecast['signals'],
                        'interpretation': forecast['interpretation']
                    })
                    
            except Exception as e:
                print(f"Warning: Error processing {region}: {e}")
        
        return alerts
    
    def generate_report(self, alerts):
        """
        Generate formatted alert report.
        
        Parameters
        ----------
        alerts : list
            List of alert dictionaries
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print("\n" + "!" * 70)
        print("QO3 SEISMIC RISK ALERT REPORT")
        print("!" * 70)
        print(f"\nTimestamp: {timestamp}")
        print(f"Regions monitored: {', '.join(self.regions)}")
        print(f"Target: M≥{self.target_mag} within {self.horizon} days")
        print(f"Alert threshold: {self.alert_threshold}")
        
        if alerts:
            print(f"\n{'!'*70}")
            print(f"⚠️  {len(alerts)} ELEVATED RISK ALERT(S) DETECTED")
            print("!" * 70)
            
            for alert in alerts:
                print(f"\n{'='*50}")
                print(f"REGION: {alert['region'].upper()}")
                print("=" * 50)
                print(f"Date: {alert['date']}")
                print(f"QO3 Score: {alert['qo3_score']:.3f}")
                print(f"Risk Level: {alert['risk_level']}")
                
                print(f"\nFIO Indicators:")
                print(f"  b-value: {alert['b_value']}")
                print(f"  CV: {alert['cv']}")
                
                print(f"\nActive Signals:")
                for signal, active in alert['signals'].items():
                    status = "✓ ACTIVE" if active else "○ inactive"
                    print(f"  {signal}: {status}")
                
                print(f"\nInterpretation:")
                print(f"  {alert['interpretation']}")
        else:
            print("\n" + "-" * 70)
            print("✓ All monitored regions at NORMAL risk levels")
            print("-" * 70)
        
        print("\n" + "=" * 70)
        print("END OF ALERT REPORT")
        print("=" * 70)
        
        return alerts


def main():
    """Run alert system demonstration."""
    
    print("=" * 70)
    print("QO3 AUTOMATED ALERT SYSTEM")
    print("=" * 70)
    
    # Initialize alert system
    alert_system = QO3AlertSystem(
        regions=['japan', 'california', 'turkey'],
        target_mag=5.5,
        horizon=7,
        alert_threshold=0.4  # Lower threshold for demo
    )
    
    print(f"\nConfiguration:")
    print(f"  Regions: {alert_system.regions}")
    print(f"  Target: M≥{alert_system.target_mag}")
    print(f"  Horizon: {alert_system.horizon} days")
    print(f"  Alert threshold: {alert_system.alert_threshold}")
    
    print("\nChecking regions...")
    
    # Check all regions
    alerts = alert_system.check_all_regions()
    
    # Generate report
    alert_system.generate_report(alerts)
    
    # Save alerts to file
    if alerts:
        output_file = 'alerts.json'
        with open(output_file, 'w') as f:
            json.dump(alerts, f, indent=2, default=str)
        print(f"\nAlerts saved to: {output_file}")


if __name__ == "__main__":
    main()
