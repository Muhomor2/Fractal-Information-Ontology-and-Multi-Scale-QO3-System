"""
FIO-QO3: Fractal Information Ontology for Seismic Risk Detection
================================================================

A physics-based framework for detecting pre-event regimes in seismic catalogs.

Author: Igor Chechelnitsky
ORCID: 0009-0007-4607-1946
License: CC BY-NC-ND 4.0 (Commercial Rights Reserved)
Version: 1.0.0

Contact: Facebook - Igor Chechelnitsky

Example Usage:
    >>> from src.qo3_system import QO3System
    >>> system = QO3System(region='japan')
    >>> system.fetch_data('2020-01-01', '2024-01-01')
    >>> system.compute_features()
    >>> forecast = system.generate_forecast(target_mag=5.0, horizon=7)
"""

__version__ = "1.0.0"
__author__ = "Igor Chechelnitsky"
__license__ = "CC BY-NC-ND 4.0"
__contact__ = "Facebook: Igor Chechelnitsky"

from .qo3_system import (
    QO3System,
    FIOFeatureEngine,
    QO3RiskModel,
    USGSDataFetcher,
    RegionConfig,
    RiskLevel,
    REGIONS,
    create_target_variable
)

from .validation import (
    TemporalValidator,
    ValidationResults,
    bootstrap_confidence_interval,
    compute_fio_contribution,
    print_validation_report
)

__all__ = [
    # Main system
    'QO3System',
    'FIOFeatureEngine', 
    'QO3RiskModel',
    'USGSDataFetcher',
    
    # Configuration
    'RegionConfig',
    'RiskLevel',
    'REGIONS',
    
    # Utilities
    'create_target_variable',
    
    # Validation
    'TemporalValidator',
    'ValidationResults',
    'bootstrap_confidence_interval',
    'compute_fio_contribution',
    'print_validation_report',
]
