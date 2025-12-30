#!/usr/bin/env python3
"""
QO3 Global Universal Seismic Risk Detection System
===================================================

Fractal Information Ontology (FIO) based earthquake regime detection.

Author: Igor Chechelnitsky
ORCID: 0009-0007-4607-1946
Location: Ashkelon, Israel
Version: 1.0.0 (December 2025)

License: MIT

Theory: This system detects pre-event regimes in seismic catalogs by analyzing
temporal structure through FIO invariants (b-value, CV, entropy) rather than
spatial clustering alone.

References:
    - Aki, K. (1965). Bull. Earthq. Res. Inst. 43, 237-239
    - Bak, P., Tang, C., Wiesenfeld, K. (1987). Phys. Rev. Lett. 59, 381-384
    - Scholz, C.H. (2015). Geophys. Res. Lett. 42, 1399-1402
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass
from enum import Enum
import warnings
import json
import requests
from io import StringIO

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

class RiskLevel(Enum):
    """Risk level classification based on QO3 score."""
    LOW = "🟢 LOW"
    MODERATE = "🟡 MODERATE"  
    ELEVATED = "🟠 ELEVATED"
    HIGH = "🔴 HIGH"

@dataclass
class RegionConfig:
    """Configuration for a seismic region."""
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    mc: float  # Magnitude of completeness
    description: str

# Pre-configured regions
REGIONS: Dict[str, RegionConfig] = {
    "global": RegionConfig("Global", -90, 90, -180, 180, 4.5, "Worldwide seismicity"),
    "japan": RegionConfig("Japan", 30, 46, 128, 148, 2.5, "Japanese archipelago"),
    "tohoku": RegionConfig("Tohoku", 35, 42, 140, 145, 2.0, "Tohoku segment, Japan"),
    "california": RegionConfig("California", 32, 42, -125, -114, 2.5, "California, USA"),
    "mediterranean": RegionConfig("Mediterranean", 30, 46, -10, 40, 3.0, "Mediterranean basin"),
    "turkey": RegionConfig("Turkey-Syria", 35, 42, 25, 45, 3.0, "Anatolian plate"),
    "chile": RegionConfig("Chile", -45, -18, -76, -68, 3.5, "Chilean subduction"),
    "indonesia": RegionConfig("Indonesia", -12, 8, 94, 142, 4.0, "Indonesian arc"),
    "iran": RegionConfig("Iran", 25, 40, 44, 64, 3.5, "Iranian plateau"),
    "himalaya": RegionConfig("Himalaya", 25, 38, 70, 100, 4.0, "Himalayan front"),
    "alaska": RegionConfig("Alaska", 50, 72, -180, -130, 3.0, "Alaska-Aleutian"),
    "newzealand": RegionConfig("New Zealand", -50, -34, 165, 180, 3.0, "New Zealand"),
    "israel": RegionConfig("Israel-DST", 29, 34, 34, 37, 2.0, "Dead Sea Transform"),
}

# ============================================================================
# DATA ACQUISITION
# ============================================================================

class USGSDataFetcher:
    """Fetches earthquake data from USGS API."""
    
    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    @staticmethod
    def fetch(region: RegionConfig, 
              start_date: str, 
              end_date: str,
              min_magnitude: float = 2.0) -> pd.DataFrame:
        """
        Fetch earthquake catalog from USGS.
        
        Parameters
        ----------
        region : RegionConfig
            Region configuration with boundaries
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        min_magnitude : float
            Minimum magnitude threshold
            
        Returns
        -------
        pd.DataFrame
            Earthquake catalog with columns: datetime, lat, lon, depth, mag
        """
        params = {
            "format": "csv",
            "starttime": start_date,
            "endtime": end_date,
            "minlatitude": region.lat_min,
            "maxlatitude": region.lat_max,
            "minlongitude": region.lon_min,
            "maxlongitude": region.lon_max,
            "minmagnitude": min_magnitude,
            "orderby": "time-asc"
        }
        
        try:
            response = requests.get(USGSDataFetcher.BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text))
            
            # Standardize columns
            df = df.rename(columns={
                'time': 'datetime',
                'latitude': 'lat',
                'longitude': 'lon',
                'mag': 'magnitude'
            })
            
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df[['datetime', 'lat', 'lon', 'depth', 'magnitude']].dropna()
            df = df.sort_values('datetime').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch USGS data: {e}")

# ============================================================================
# FIO FEATURE COMPUTATION
# ============================================================================

class FIOFeatureEngine:
    """
    Fractal Information Ontology feature computation engine.
    
    Computes the core FIO invariants:
    - b-value (Aki-Utsu estimator)
    - CV of inter-event intervals
    - Shannon entropy of interval distribution
    - Rate dynamics (multi-scale)
    """
    
    def __init__(self, mc: float = 2.0):
        """
        Initialize feature engine.
        
        Parameters
        ----------
        mc : float
            Magnitude of completeness threshold
        """
        self.mc = mc
    
    def compute_b_value(self, magnitudes: np.ndarray) -> float:
        """
        Compute b-value using Aki-Utsu maximum likelihood estimator.
        
        b = log10(e) / (M_mean - Mc)
        
        Parameters
        ----------
        magnitudes : np.ndarray
            Array of magnitudes >= Mc
            
        Returns
        -------
        float
            Estimated b-value
        """
        mags = magnitudes[magnitudes >= self.mc]
        if len(mags) < 10:
            return np.nan
        
        mean_mag = np.mean(mags)
        if mean_mag <= self.mc:
            return np.nan
            
        b = np.log10(np.e) / (mean_mag - self.mc)
        
        # Physical bounds
        return np.clip(b, 0.3, 2.5)
    
    def compute_cv(self, intervals: np.ndarray) -> float:
        """
        Compute coefficient of variation of inter-event intervals.
        
        CV = σ(τ) / μ(τ)
        
        Parameters
        ----------
        intervals : np.ndarray
            Array of inter-event intervals (in seconds or hours)
            
        Returns
        -------
        float
            Coefficient of variation
        """
        intervals = intervals[intervals > 0]
        if len(intervals) < 5:
            return np.nan
            
        mean_int = np.mean(intervals)
        if mean_int == 0:
            return np.nan
            
        cv = np.std(intervals) / mean_int
        return np.clip(cv, 0.01, 10.0)
    
    def compute_entropy(self, intervals: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute Shannon entropy of interval distribution.
        
        S(W) = -Σ p_k * log2(p_k)
        
        Parameters
        ----------
        intervals : np.ndarray
            Array of inter-event intervals
        n_bins : int
            Number of histogram bins
            
        Returns
        -------
        float
            Shannon entropy in bits
        """
        intervals = intervals[intervals > 0]
        if len(intervals) < n_bins:
            return np.nan
        
        # Create histogram
        hist, _ = np.histogram(intervals, bins=n_bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        if len(hist) == 0:
            return 0.0
            
        # Normalize to probability
        prob = hist / hist.sum()
        
        # Shannon entropy
        entropy = -np.sum(prob * np.log2(prob))
        return entropy
    
    def compute_daily_features(self, 
                               catalog: pd.DataFrame,
                               b_window: int = 30,
                               cv_window: int = 14,
                               rate_windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """
        Compute daily FIO features from earthquake catalog.
        
        Parameters
        ----------
        catalog : pd.DataFrame
            Earthquake catalog with columns: datetime, magnitude
        b_window : int
            Rolling window for b-value computation (days)
        cv_window : int
            Rolling window for CV computation (days)
        rate_windows : list
            Windows for rate dynamics computation
            
        Returns
        -------
        pd.DataFrame
            Daily feature matrix
        """
        # Create daily aggregation
        catalog = catalog.copy()
        catalog['date'] = catalog['datetime'].dt.date
        
        # Get date range
        date_range = pd.date_range(
            start=catalog['date'].min(),
            end=catalog['date'].max(),
            freq='D'
        )
        
        features = []
        
        for current_date in date_range:
            current_date_py = current_date.date()
            
            # === Rate dynamics ===
            rates = {}
            for window in rate_windows:
                window_start = current_date_py - timedelta(days=window)
                mask = (catalog['date'] > window_start) & (catalog['date'] <= current_date_py)
                rates[f'rate_{window}d'] = mask.sum()
            
            # === b-value ===
            b_start = current_date_py - timedelta(days=b_window)
            b_mask = (catalog['date'] > b_start) & (catalog['date'] <= current_date_py)
            b_mags = catalog.loc[b_mask, 'magnitude'].values
            b_value = self.compute_b_value(b_mags)
            
            # === CV of inter-event intervals ===
            cv_start = current_date_py - timedelta(days=cv_window)
            cv_mask = (catalog['date'] > cv_start) & (catalog['date'] <= current_date_py)
            cv_times = catalog.loc[cv_mask, 'datetime'].values
            
            if len(cv_times) > 2:
                intervals = np.diff(cv_times).astype('timedelta64[s]').astype(float)
                cv_value = self.compute_cv(intervals)
                entropy_value = self.compute_entropy(intervals)
            else:
                cv_value = np.nan
                entropy_value = np.nan
            
            # === Max magnitude in window ===
            max_mag_7d_start = current_date_py - timedelta(days=7)
            max_mag_mask = (catalog['date'] > max_mag_7d_start) & (catalog['date'] <= current_date_py)
            max_mag_7d = catalog.loc[max_mag_mask, 'magnitude'].max() if max_mag_mask.sum() > 0 else np.nan
            
            features.append({
                'date': current_date_py,
                **rates,
                'b_value': b_value,
                'cv': cv_value,
                'entropy': entropy_value,
                'max_mag_7d': max_mag_7d
            })
        
        df = pd.DataFrame(features)
        
        # === Compute gradients (7-day changes) ===
        df['b_value_change_7d'] = df['b_value'].diff(7)
        df['cv_change_7d'] = df['cv'].diff(7)
        df['entropy_change_7d'] = df['entropy'].diff(7)
        df['rate_change_7d'] = df['rate_7d'].diff(7)
        
        # === Rate ratios ===
        df['rate_ratio_7_30'] = df['rate_7d'] / df['rate_30d'].replace(0, np.nan)
        
        # === Distance from critical CV ===
        df['cv_distance_from_1'] = np.abs(df['cv'] - 1.0)
        
        # === Seismic Information Deficit (SID) ===
        background_entropy = df['entropy'].rolling(90, min_periods=30).mean()
        df['sid'] = 1 - (df['entropy'] / background_entropy.replace(0, np.nan))
        
        return df

# ============================================================================
# QO3 RISK SCORE COMPUTATION
# ============================================================================

class QO3RiskModel:
    """
    QO3 Risk Score computation model.
    
    Combines FIO features into an integrated risk score using
    physically-motivated thresholds calibrated from training data.
    """
    
    # Default thresholds (calibrated on Tohoku 2017-2021)
    DEFAULT_THRESHOLDS = {
        'b_critical': 0.75,      # 15th percentile of background
        'cv_critical_range': 0.2,  # ±0.2 from CV=1
        'rate_acceleration': 1.5,  # 7d/30d ratio threshold
        'entropy_drop': 0.3        # SID threshold
    }
    
    def __init__(self, thresholds: Optional[Dict] = None):
        """
        Initialize QO3 model.
        
        Parameters
        ----------
        thresholds : dict, optional
            Custom thresholds for risk computation
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
    
    def compute_risk_score(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Compute QO3 risk score from FIO features.
        
        Risk components:
        - Stress signal: b < b_critical
        - Sync signal: |CV - 1| < cv_critical_range
        - Energy trigger: rate acceleration > threshold
        - Information compression: SID > entropy_drop
        
        Parameters
        ----------
        features : pd.DataFrame
            Daily feature matrix from FIOFeatureEngine
            
        Returns
        -------
        pd.DataFrame
            Features with added risk score columns
        """
        df = features.copy()
        
        # === Component signals ===
        
        # Stress accumulation signal
        df['stress_signal'] = (df['b_value'] < self.thresholds['b_critical']).astype(float)
        
        # Synchronization signal (CV approaching 1)
        df['sync_signal'] = (df['cv_distance_from_1'] < self.thresholds['cv_critical_range']).astype(float)
        
        # Energy acceleration signal
        df['energy_signal'] = (df['rate_ratio_7_30'] > self.thresholds['rate_acceleration']).astype(float)
        
        # Information compression signal
        df['compression_signal'] = (df['sid'] > self.thresholds['entropy_drop']).astype(float)
        
        # === Weighted QO3 score ===
        # Weights based on feature importance from ML validation
        weights = {
            'stress': 0.35,
            'sync': 0.25,
            'energy': 0.25,
            'compression': 0.15
        }
        
        df['qo3_score'] = (
            weights['stress'] * df['stress_signal'] +
            weights['sync'] * df['sync_signal'] +
            weights['energy'] * df['energy_signal'] +
            weights['compression'] * df['compression_signal'].fillna(0)
        )
        
        # === Risk level classification ===
        df['risk_level'] = df['qo3_score'].apply(self._classify_risk)
        
        return df
    
    @staticmethod
    def _classify_risk(score: float) -> str:
        """Classify risk level based on QO3 score."""
        if pd.isna(score):
            return RiskLevel.LOW.value
        elif score < 0.3:
            return RiskLevel.LOW.value
        elif score < 0.5:
            return RiskLevel.MODERATE.value
        elif score < 0.7:
            return RiskLevel.ELEVATED.value
        else:
            return RiskLevel.HIGH.value

# ============================================================================
# TARGET VARIABLE CONSTRUCTION
# ============================================================================

def create_target_variable(catalog: pd.DataFrame,
                           features: pd.DataFrame,
                           target_mag: float = 5.0,
                           horizon: int = 7) -> pd.DataFrame:
    """
    Create strictly future-looking target variable.
    
    Y_t = 1 if max(M_{t+1}, ..., M_{t+horizon}) >= target_mag
    
    CRITICAL: This ensures no data leakage - target is computed
    only from events AFTER the current date.
    
    Parameters
    ----------
    catalog : pd.DataFrame
        Earthquake catalog
    features : pd.DataFrame
        Daily feature matrix
    target_mag : float
        Target magnitude threshold
    horizon : int
        Forecast horizon in days
        
    Returns
    -------
    pd.DataFrame
        Features with target column added
    """
    catalog = catalog.copy()
    catalog['date'] = pd.to_datetime(catalog['datetime']).dt.date
    
    features = features.copy()
    targets = []
    
    for idx, row in features.iterrows():
        current_date = row['date']
        
        # STRICTLY FUTURE: from t+1 to t+horizon
        future_start = current_date + timedelta(days=1)
        future_end = current_date + timedelta(days=horizon)
        
        future_mask = (catalog['date'] >= future_start) & (catalog['date'] <= future_end)
        future_events = catalog[future_mask]
        
        if len(future_events) > 0:
            max_future_mag = future_events['magnitude'].max()
            target = 1 if max_future_mag >= target_mag else 0
        else:
            target = 0
            
        targets.append(target)
    
    features['target'] = targets
    return features

# ============================================================================
# MAIN SYSTEM CLASS
# ============================================================================

class QO3System:
    """
    Main QO3 Universal Seismic Risk Detection System.
    
    This class provides the complete pipeline:
    1. Data acquisition (USGS API)
    2. FIO feature computation
    3. QO3 risk score calculation
    4. Forecast generation
    
    Example
    -------
    >>> system = QO3System(region='tohoku')
    >>> system.fetch_data('2020-01-01', '2024-01-01')
    >>> system.compute_features()
    >>> forecast = system.generate_forecast(target_mag=5.0, horizon=7)
    >>> print(forecast.current_risk)
    """
    
    def __init__(self, 
                 region: Union[str, RegionConfig] = 'japan',
                 mc: Optional[float] = None):
        """
        Initialize QO3 system.
        
        Parameters
        ----------
        region : str or RegionConfig
            Region name or custom configuration
        mc : float, optional
            Override magnitude of completeness
        """
        if isinstance(region, str):
            if region not in REGIONS:
                raise ValueError(f"Unknown region '{region}'. Available: {list(REGIONS.keys())}")
            self.region = REGIONS[region]
        else:
            self.region = region
            
        self.mc = mc or self.region.mc
        self.catalog: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.feature_engine = FIOFeatureEngine(mc=self.mc)
        self.risk_model = QO3RiskModel()
        
    def fetch_data(self, 
                   start_date: str, 
                   end_date: str,
                   min_magnitude: Optional[float] = None) -> pd.DataFrame:
        """
        Fetch earthquake catalog from USGS.
        
        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        min_magnitude : float, optional
            Minimum magnitude (defaults to Mc - 0.5)
            
        Returns
        -------
        pd.DataFrame
            Earthquake catalog
        """
        min_mag = min_magnitude or max(self.mc - 0.5, 1.0)
        
        print(f"Fetching data for {self.region.name}...")
        print(f"  Bounds: {self.region.lat_min}°-{self.region.lat_max}°N, "
              f"{self.region.lon_min}°-{self.region.lon_max}°E")
        print(f"  Period: {start_date} to {end_date}")
        print(f"  Min magnitude: {min_mag}")
        
        self.catalog = USGSDataFetcher.fetch(
            self.region, start_date, end_date, min_mag
        )
        
        print(f"  Retrieved {len(self.catalog)} events")
        return self.catalog
    
    def load_catalog(self, catalog: pd.DataFrame) -> None:
        """Load pre-existing catalog."""
        required_cols = ['datetime', 'magnitude']
        if not all(col in catalog.columns for col in required_cols):
            raise ValueError(f"Catalog must have columns: {required_cols}")
        self.catalog = catalog.copy()
        
    def compute_features(self) -> pd.DataFrame:
        """
        Compute FIO features and QO3 risk scores.
        
        Returns
        -------
        pd.DataFrame
            Daily feature matrix with risk scores
        """
        if self.catalog is None:
            raise RuntimeError("No catalog loaded. Call fetch_data() first.")
        
        print("Computing FIO features...")
        self.features = self.feature_engine.compute_daily_features(self.catalog)
        
        print("Computing QO3 risk scores...")
        self.features = self.risk_model.compute_risk_score(self.features)
        
        print(f"  Computed {len(self.features)} daily observations")
        return self.features
    
    def generate_forecast(self,
                          target_mag: float = 5.0,
                          horizon: int = 7) -> Dict:
        """
        Generate current risk forecast.
        
        Parameters
        ----------
        target_mag : float
            Target magnitude for forecast
        horizon : int
            Forecast horizon in days
            
        Returns
        -------
        dict
            Forecast report with current risk assessment
        """
        if self.features is None:
            raise RuntimeError("No features computed. Call compute_features() first.")
        
        # Get latest observation
        latest = self.features.iloc[-1]
        
        # Compute historical baseline
        baseline_rate = (self.catalog['magnitude'] >= target_mag).mean()
        
        forecast = {
            'region': self.region.name,
            'date': str(latest['date']),
            'target_magnitude': target_mag,
            'horizon_days': horizon,
            'current_assessment': {
                'qo3_score': round(latest['qo3_score'], 3) if pd.notna(latest['qo3_score']) else None,
                'risk_level': latest['risk_level'],
                'b_value': round(latest['b_value'], 3) if pd.notna(latest['b_value']) else None,
                'cv': round(latest['cv'], 3) if pd.notna(latest['cv']) else None,
                'entropy': round(latest['entropy'], 3) if pd.notna(latest['entropy']) else None,
            },
            'signals': {
                'stress_accumulation': bool(latest['stress_signal']),
                'temporal_synchronization': bool(latest['sync_signal']),
                'energy_acceleration': bool(latest['energy_signal']),
            },
            'baseline_probability': round(baseline_rate * 100, 1),
            'interpretation': self._interpret_forecast(latest, target_mag, horizon)
        }
        
        return forecast
    
    @staticmethod
    def _interpret_forecast(latest: pd.Series, target_mag: float, horizon: int) -> str:
        """Generate human-readable forecast interpretation."""
        score = latest['qo3_score']
        
        if pd.isna(score) or score < 0.3:
            return (f"Background seismicity regime. No elevated risk indicators "
                    f"for M≥{target_mag} within {horizon} days.")
        elif score < 0.5:
            return (f"Moderate anomaly detected. Some FIO indicators suggest "
                    f"departure from background regime. Enhanced monitoring recommended.")
        elif score < 0.7:
            return (f"Elevated risk regime. Multiple FIO indicators in pre-critical state. "
                    f"Probability of M≥{target_mag} within {horizon} days is statistically elevated.")
        else:
            return (f"High-risk regime detected. System approaching critical state. "
                    f"Strong FIO signature suggests elevated probability of significant event.")

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for QO3 system."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="QO3 Universal Seismic Risk Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qo3_system.py --region tohoku --target_mag 5.0 --horizon 7
  python qo3_system.py --region california --target_mag 4.5 --horizon 14
  python qo3_system.py --region global --target_mag 6.0 --horizon 30

Available regions: global, japan, tohoku, california, mediterranean, 
                   turkey, chile, indonesia, iran, himalaya, alaska, 
                   newzealand, israel
        """
    )
    
    parser.add_argument('--region', type=str, default='japan',
                        help='Region name (default: japan)')
    parser.add_argument('--target_mag', type=float, default=5.0,
                        help='Target magnitude (default: 5.0)')
    parser.add_argument('--horizon', type=int, default=7,
                        help='Forecast horizon in days (default: 7)')
    parser.add_argument('--start_date', type=str, default=None,
                        help='Start date (default: 2 years ago)')
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date (default: today)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file (optional)')
    
    args = parser.parse_args()
    
    # Set date defaults
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    if args.start_date is None:
        args.start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    # Run system
    print("=" * 60)
    print("QO3 UNIVERSAL SEISMIC RISK DETECTION SYSTEM")
    print("Fractal Information Ontology (FIO) Framework")
    print("=" * 60)
    print()
    
    try:
        system = QO3System(region=args.region)
        system.fetch_data(args.start_date, args.end_date)
        system.compute_features()
        forecast = system.generate_forecast(args.target_mag, args.horizon)
        
        print()
        print("=" * 60)
        print("CURRENT RISK ASSESSMENT")
        print("=" * 60)
        print(f"Region: {forecast['region']}")
        print(f"Date: {forecast['date']}")
        print(f"Target: M≥{forecast['target_magnitude']} within {forecast['horizon_days']} days")
        print()
        print(f"Risk Level: {forecast['current_assessment']['risk_level']}")
        print(f"QO3 Score: {forecast['current_assessment']['qo3_score']}")
        print()
        print("FIO Indicators:")
        print(f"  b-value: {forecast['current_assessment']['b_value']}")
        print(f"  CV: {forecast['current_assessment']['cv']}")
        print(f"  Entropy: {forecast['current_assessment']['entropy']}")
        print()
        print("Active Signals:")
        print(f"  Stress accumulation: {forecast['signals']['stress_accumulation']}")
        print(f"  Temporal sync: {forecast['signals']['temporal_synchronization']}")
        print(f"  Energy acceleration: {forecast['signals']['energy_acceleration']}")
        print()
        print(f"Baseline probability: {forecast['baseline_probability']}%")
        print()
        print("Interpretation:")
        print(f"  {forecast['interpretation']}")
        print()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(forecast, f, indent=2)
            print(f"Forecast saved to: {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
