# FIO-QO3: Fractal Information Ontology for Seismic Risk Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**A physics-based framework for detecting pre-event regimes in seismic catalogs**

## Overview

FIO-QO3 is a seismic risk detection system based on Fractal Information Ontology (FIO). Unlike traditional approaches focused on spatial clustering (ETAS), FIO-QO3 analyzes the **temporal structure** of earthquake sequences to detect **regime changes** preceding significant events.

**Key Innovation**: FIO treats seismicity as a fractal information field where pre-event preparation manifests as measurable changes in b-value, inter-event interval statistics (CV), and information entropy.

### Scientific Positioning

- **NOT** earthquake prediction → **Regime detection**
- **NOT** deterministic forecast → **Probabilistic risk stratification**  
- **NOT** time/location/magnitude prediction → **Elevated vulnerability states**

## Theory

### Fractal Information Ontology (FIO)

FIO postulates that seismicity is a manifestation of Self-Organized Criticality (SOC) with long-term memory and phase transitions.

**Core Invariants**:

1. **b-value** (Aki-Utsu): `b = log₁₀(e) / (M̄ - Mc)`
   - Tracks stress accumulation
   - b < 0.8 indicates pre-event regime

2. **CV** (Coefficient of Variation): `CV = σ(Δt) / μ(Δt)`
   - Measures temporal clustering
   - CV → 1 signals approach to critical state

3. **Entropy**: `S(W) = -Σ pₖ log₂(pₖ)`
   - Information compression measure
   - Decrease indicates pre-event focusing

### Target Definition (Strictly Future)

```
Y_t = 𝟙[max(M_{t+1}, ..., M_{t+H}) ≥ M_target]
```

**Critical**: Target computed only from events AFTER current date. No data leakage.

## Installation

```bash
# Clone repository
git clone https://github.com/ichechelnitsky/FIO-QO3.git
cd FIO-QO3

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m src.qo3_system --help
```

### Requirements

- Python 3.8+
- numpy >= 1.20
- pandas >= 1.3
- requests >= 2.25
- scikit-learn >= 1.0 (optional, for ML validation)

## Quick Start

### Command Line Interface

```bash
# Current risk assessment for Japan
python -m src.qo3_system --region japan --target_mag 5.0 --horizon 7

# California with 14-day horizon
python -m src.qo3_system --region california --target_mag 4.5 --horizon 14

# Global M6+ assessment
python -m src.qo3_system --region global --target_mag 6.0 --horizon 30

# Save output to JSON
python -m src.qo3_system --region tohoku --output forecast.json
```

### Python API - Basic Usage

```python
from src.qo3_system import QO3System

# Initialize system for Japan
system = QO3System(region='japan')

# Fetch 2 years of data from USGS
system.fetch_data('2023-01-01', '2025-01-01')

# Compute FIO features
system.compute_features()

# Generate current forecast
forecast = system.generate_forecast(target_mag=5.0, horizon=7)

# Display results
print(f"Risk Level: {forecast['current_assessment']['risk_level']}")
print(f"QO3 Score: {forecast['current_assessment']['qo3_score']}")
print(f"b-value: {forecast['current_assessment']['b_value']}")
print(f"CV: {forecast['current_assessment']['cv']}")
```

## Usage Examples

### Example 1: Real-Time Monitoring Dashboard

```python
"""
Real-time seismic risk monitoring for multiple regions.
"""
from src.qo3_system import QO3System, REGIONS
import json
from datetime import datetime, timedelta

def monitor_all_regions():
    """Generate risk report for all configured regions."""
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    reports = {}
    
    for region_name in ['japan', 'california', 'turkey', 'chile']:
        print(f"\n{'='*50}")
        print(f"Processing: {region_name.upper()}")
        print('='*50)
        
        try:
            system = QO3System(region=region_name)
            system.fetch_data(start_date, end_date)
            system.compute_features()
            
            forecast = system.generate_forecast(target_mag=5.5, horizon=14)
            reports[region_name] = forecast
            
            print(f"  Risk Level: {forecast['current_assessment']['risk_level']}")
            print(f"  QO3 Score: {forecast['current_assessment']['qo3_score']}")
            
        except Exception as e:
            print(f"  Error: {e}")
            reports[region_name] = {'error': str(e)}
    
    # Save consolidated report
    with open('global_risk_report.json', 'w') as f:
        json.dump(reports, f, indent=2, default=str)
    
    return reports

if __name__ == "__main__":
    monitor_all_regions()
```

### Example 2: Custom Region Analysis

```python
"""
Analyze a custom region not in the predefined list.
"""
from src.qo3_system import QO3System, RegionConfig

# Define custom region: Eastern Mediterranean
custom_region = RegionConfig(
    name="Eastern Mediterranean",
    lat_min=32.0,
    lat_max=42.0,
    lon_min=20.0,
    lon_max=40.0,
    mc=3.5,  # Magnitude of completeness
    description="Eastern Mediterranean including Greece, Turkey, Cyprus"
)

# Initialize with custom region
system = QO3System(region=custom_region)

# Fetch and analyze
system.fetch_data('2022-01-01', '2025-01-01')
system.compute_features()

# Get forecast
forecast = system.generate_forecast(target_mag=5.0, horizon=7)

print(f"\n{custom_region.name} Risk Assessment")
print(f"Date: {forecast['date']}")
print(f"Risk: {forecast['current_assessment']['risk_level']}")
print(f"Interpretation: {forecast['interpretation']}")
```

### Example 3: Historical Validation (Backtesting)

```python
"""
Validate QO3 performance on historical data with train/test split.
"""
from src.qo3_system import QO3System, create_target_variable
from src.validation import TemporalValidator, print_validation_report

# Setup
system = QO3System(region='tohoku')
system.fetch_data('2015-01-01', '2024-01-01')
features = system.compute_features()

# Create target: M≥5.0 within 7 days
features_with_target = create_target_variable(
    system.catalog, 
    features, 
    target_mag=5.0, 
    horizon=7
)

# Validate with temporal split
validator = TemporalValidator(train_end_date='2021-12-31')
train_df, test_df = validator.prepare_data(features_with_target)

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Train and evaluate
validator.train(train_df)
results = validator.validate(test_df)

# Print report
print_validation_report(
    results,
    train_period="2015-2021",
    test_period="2022-2023"
)
```

### Example 4: Feature Analysis and Visualization

```python
"""
Analyze FIO feature distributions and correlations.
"""
from src.qo3_system import QO3System
import pandas as pd

system = QO3System(region='japan')
system.fetch_data('2020-01-01', '2024-01-01')
features = system.compute_features()

# Feature statistics
print("\n=== FIO Feature Statistics ===\n")
fio_cols = ['b_value', 'cv', 'entropy', 'qo3_score']
print(features[fio_cols].describe())

# Risk level distribution
print("\n=== Risk Level Distribution ===\n")
print(features['risk_level'].value_counts())

# Correlation with high-risk periods
print("\n=== Feature Correlations ===\n")
high_risk = features['qo3_score'] > 0.5
print(f"High risk days: {high_risk.sum()} ({high_risk.mean()*100:.1f}%)")

# Export for external analysis
features.to_csv('qo3_features_japan.csv', index=False)
print("\nFeatures exported to qo3_features_japan.csv")
```

### Example 5: Automated Alert System

```python
"""
Simple alert system that checks risk levels and sends notifications.
"""
from src.qo3_system import QO3System
from datetime import datetime

def check_and_alert(regions=['japan', 'california'], 
                    alert_threshold=0.6):
    """
    Check risk levels and generate alerts for high-risk regions.
    """
    alerts = []
    
    for region in regions:
        system = QO3System(region=region)
        
        # Fetch recent data (last 6 months is usually sufficient)
        system.fetch_data('2024-07-01', datetime.now().strftime('%Y-%m-%d'))
        system.compute_features()
        
        forecast = system.generate_forecast(target_mag=5.5, horizon=7)
        score = forecast['current_assessment']['qo3_score']
        
        if score and score >= alert_threshold:
            alerts.append({
                'region': region,
                'score': score,
                'level': forecast['current_assessment']['risk_level'],
                'signals': forecast['signals'],
                'interpretation': forecast['interpretation']
            })
    
    # Process alerts
    if alerts:
        print("\n" + "!"*60)
        print("ELEVATED RISK ALERTS")
        print("!"*60)
        for alert in alerts:
            print(f"\nRegion: {alert['region'].upper()}")
            print(f"Score: {alert['score']:.3f}")
            print(f"Level: {alert['level']}")
            print(f"Active signals:")
            for sig, active in alert['signals'].items():
                if active:
                    print(f"  ✓ {sig}")
    else:
        print("\nAll monitored regions at normal risk levels.")
    
    return alerts

if __name__ == "__main__":
    check_and_alert()
```

### Example 6: Comparative Analysis Across Regions

```python
"""
Compare QO3 performance metrics across different tectonic settings.
"""
from src.qo3_system import QO3System, create_target_variable
from src.validation import TemporalValidator
import pandas as pd

regions_to_compare = {
    'tohoku': {'mag': 5.0, 'horizon': 7},
    'california': {'mag': 4.5, 'horizon': 7},
    'chile': {'mag': 5.5, 'horizon': 14},
    'turkey': {'mag': 5.0, 'horizon': 7}
}

results_summary = []

for region, params in regions_to_compare.items():
    print(f"\nProcessing {region}...")
    
    try:
        system = QO3System(region=region)
        system.fetch_data('2018-01-01', '2024-01-01')
        features = system.compute_features()
        
        features_with_target = create_target_variable(
            system.catalog, features,
            target_mag=params['mag'],
            horizon=params['horizon']
        )
        
        validator = TemporalValidator(train_end_date='2022-06-30')
        train_df, test_df = validator.prepare_data(features_with_target)
        
        if len(train_df) > 100 and len(test_df) > 30:
            validator.train(train_df)
            val_results = validator.validate(test_df)
            
            results_summary.append({
                'Region': region,
                'Target_Mag': params['mag'],
                'Horizon': params['horizon'],
                'Baseline_Rate': f"{val_results.baseline_rate:.1%}",
                'PR_AUC': f"{val_results.pr_auc:.3f}",
                'Skill_Score': f"{val_results.skill_score:.3f}",
                'N_Test': val_results.n_test
            })
    except Exception as e:
        print(f"  Error: {e}")

# Display comparison table
df_results = pd.DataFrame(results_summary)
print("\n" + "="*70)
print("CROSS-REGIONAL COMPARISON")
print("="*70)
print(df_results.to_string(index=False))
```

## Available Regions

| Region | Bounds | Mc | Description |
|--------|--------|-----|-------------|
| `global` | -90°–90°N, -180°–180°E | 4.5 | Worldwide |
| `japan` | 30°–46°N, 128°–148°E | 2.5 | Japanese archipelago |
| `tohoku` | 35°–42°N, 140°–145°E | 2.0 | Tohoku segment |
| `california` | 32°–42°N, 125°–114°W | 2.5 | California |
| `mediterranean` | 30°–46°N, 10°W–40°E | 3.0 | Mediterranean basin |
| `turkey` | 35°–42°N, 25°–45°E | 3.0 | Anatolian plate |
| `chile` | 45°–18°S, 76°–68°W | 3.5 | Chilean subduction |
| `indonesia` | 12°S–8°N, 94°–142°E | 4.0 | Indonesian arc |
| `iran` | 25°–40°N, 44°–64°E | 3.5 | Iranian plateau |
| `himalaya` | 25°–38°N, 70°–100°E | 4.0 | Himalayan front |
| `alaska` | 50°–72°N, 180°–130°W | 3.0 | Alaska-Aleutian |
| `newzealand` | 50°–34°S, 165°–180°E | 3.0 | New Zealand |
| `israel` | 29°–34°N, 34°–37°E | 2.0 | Dead Sea Transform |

## Validation Results

### Tohoku, Japan (2017-2023)

| Metric | Baseline | QO3 |
|--------|----------|-----|
| Baseline rate | 39% | — |
| Precision @50% recall | 39% | 44% |
| PR-AUC | 0.39 | 0.45 |
| ROC-AUC | — | 0.56-0.87 |
| Skill Score | 0.00 | 0.08-0.10 |

**FIO Component Contribution**: ~50%

### Feature Importance

| Feature | Importance |
|---------|------------|
| b-value | 17-21% |
| CV | 13-18% |
| b-value change (7d) | 10-13% |
| CV change (7d) | 8-11% |
| Rate dynamics | 30-40% |

## Risk Level Interpretation

| QO3 Score | Level | Interpretation |
|-----------|-------|----------------|
| 0.0–0.3 | 🟢 LOW | Background regime, normal operations |
| 0.3–0.5 | 🟡 MODERATE | Some anomaly, enhanced monitoring |
| 0.5–0.7 | 🟠 ELEVATED | Pre-critical indicators, review plans |
| 0.7–1.0 | 🔴 HIGH | Critical regime, operational readiness |

## Theoretical Framework

### Theorem 1: Non-triviality
PR-AUC exceeding baseline proves non-Poissonian structure in seismic flow.

### Theorem 2: Fractal Source
Long-term memory (H > 0.5) makes b-value/CV changes informative about regime transitions.

### Theorem 3: QO3 Consistency  
Weighted score is valid regime detector when features have mutual information with target.

### Theorem 4: Impossibility
Deterministic prediction of (time, magnitude) remains impossible even with fractal memory.

## Limitations

1. **No spatial localization**: Only temporal probability within defined region
2. **Regional calibration required**: Thresholds may need adjustment
3. **Catalog completeness dependent**: Quality depends on Mc estimation
4. **Regime detection, not prediction**: Cannot specify exact event parameters

## Citation

```bibtex
@software{chechelnitsky2025fioqo3,
  author       = {Chechelnitsky, Igor},
  title        = {FIO-QO3: Fractal Information Ontology for Seismic Risk Detection},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://github.com/ichechelnitsky/FIO-QO3}
}
```

## References

- Aki, K. (1965). Maximum likelihood estimate of b. *Bull. Earthq. Res. Inst.* 43, 237-239.
- Bak, P., Tang, C., Wiesenfeld, K. (1987). Self-organized criticality. *Phys. Rev. Lett.* 59, 381-384.
- Ogata, Y. (1988). Statistical models for earthquake occurrences. *J. Am. Stat. Assoc.* 83, 9-27.
- Scholz, C.H. (2015). Stress dependence of earthquake b value. *Geophys. Res. Lett.* 42, 1399-1402.
- Sornette, D. (2006). *Critical Phenomena in Natural Sciences*. Springer.

## License

**CC BY-NC-ND 4.0 (Commercial Rights Reserved)**

Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International.

All commercial usage, deployment, and sublicensing rights are strictly reserved 
by the Author, Igor Chechelnitsky. Unauthorized commercial use is prohibited.

See [LICENSE](LICENSE) for full terms.

## Author

**Igor Chechelnitsky**  
Independent Researcher, Ashkelon, Israel  
ORCID: [0009-0007-4607-1946](https://orcid.org/0009-0007-4607-1946)

### Contact

**Facebook**: [Igor Chechelnitsky](https://www.facebook.com/igor.chechelnitsky)

For commercial licensing inquiries, collaboration proposals, or research 
partnerships, please contact via Facebook.

## Acknowledgments

Verification assistance from Claude (Anthropic), Gemini (Google), and ChatGPT (OpenAI).

---

**Disclaimer**: This system provides statistical risk indicators, not deterministic predictions. Always consult official seismological agencies for emergency decisions.
