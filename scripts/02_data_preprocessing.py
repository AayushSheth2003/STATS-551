# ============================================================================
# Script 02: Data Preprocessing for Bayesian Models
# Bayesian Bike Sharing Analysis
# ============================================================================

import pandas as pd
import numpy as np
import pickle
import os

# Get script directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, "output", "models")
os.makedirs(output_dir, exist_ok=True)

# Load data
day_data = pd.read_csv(os.path.join(project_root, "data", "day.csv"))
day_data['dteday'] = pd.to_datetime(day_data['dteday'])

# ============================================================================
# Prepare data for Stan models
# ============================================================================

# Standardize continuous predictors (for better MCMC mixing)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
day_data['temp_std'] = scaler.fit_transform(day_data[['temp']]).flatten()
day_data['atemp_std'] = scaler.fit_transform(day_data[['atemp']]).flatten()
day_data['hum_std'] = scaler.fit_transform(day_data[['hum']]).flatten()
day_data['windspeed_std'] = scaler.fit_transform(day_data[['windspeed']]).flatten()

# Create group indices (for hierarchical models)
# Season: 1=Spring, 2=Summer, 3=Fall, 4=Winter
day_data['season_idx'] = day_data['season']

# Weather situation groups
day_data['weather_idx'] = day_data['weathersit']

# Month groups (for potential hierarchical modeling)
day_data['month_idx'] = day_data['mnth']

# ============================================================================
# Data for Model 1: Hierarchical Normal Model by Season
# ============================================================================

data_hierarchical = {
    'N': len(day_data),
    'K': 4,  # 4 seasons
    'y': day_data['cnt'].values,
    'group': day_data['season_idx'].values
}

print("=== Hierarchical Model Data ===")
print(f"N (observations): {data_hierarchical['N']}")
print(f"K (groups/seasons): {data_hierarchical['K']}")
print("Group sizes:")
print(pd.Series(data_hierarchical['group']).value_counts().sort_index())

# ============================================================================
# Data for Model 2: Hierarchical Negative Binomial Regression
# (Uses same data structure as hierarchical regression)
# ============================================================================

# Select predictors for regression
# X matrix: temp, atemp, hum, windspeed, holiday, workingday, weathersit
X_regression = day_data[['temp_std', 'atemp_std', 'hum_std', 
                         'windspeed_std', 'holiday', 'workingday', 
                         'weathersit']].values

data_regression = {
    'N': len(day_data),
    'P': X_regression.shape[1],
    'y': day_data['cnt'].values,
    'X': X_regression
}

# Also prepare hierarchical regression data (same predictors, but with groups)
data_hierarchical_regression = {
    'N': len(day_data),
    'K': 4,  # 4 seasons
    'P': X_regression.shape[1],
    'y': day_data['cnt'].values,
    'X': X_regression,
    'group': day_data['season_idx'].values
}

print("\n=== Regression Model Data ===")
print(f"N (observations): {data_regression['N']}")
print(f"P (predictors): {data_regression['P']}")
print("Predictors:", ['temp_std', 'atemp_std', 'hum_std', 
                      'windspeed_std', 'holiday', 'workingday', 'weathersit'])

# ============================================================================
# Data for Model 3: Bayesian ARIMA / SARIMAX (Time Series, Optional)
# ============================================================================

# Sort data by date for time series
day_data_sorted = day_data.sort_values('dteday').reset_index(drop=True)

# Create trend variable (day index)
day_data_sorted['day_index'] = np.arange(len(day_data_sorted))

# Data for SARIMA (can be extended to SARIMAX with exogenous regressors)
data_sarima = {
    'N': len(day_data_sorted),
    'y': day_data_sorted['cnt'].values.astype(int),  # Must be integer for Negative Binomial
    'x_trend': day_data_sorted['day_index'].values.astype(float),
    'S': 7,  # Weekly seasonal pattern (7 days)
    'N_predict': 7  # Forecast 7 days ahead
}

# Optional: Prepare data for SARIMAX (with exogenous regressors)
# This would include weather and calendar variables as external inputs
# Currently not used in the active workflow
data_sarimax = None  # Placeholder for SARIMAX data structure
# If implementing SARIMAX, would include:
# - X: matrix of exogenous regressors (temp, hum, windspeed, holiday, etc.)
# - Same time series structure as SARIMA but with external covariates

print("\n=== Bayesian ARIMA / SARIMAX Model Data (Optional) ===")
print(f"N (observations): {data_sarima['N']}")
print(f"Seasonal period (S): {data_sarima['S']} (weekly pattern)")
print(f"Date range: {day_data_sorted['dteday'].min()} to {day_data_sorted['dteday'].max()}")
print("Note: SARIMAX model is optional and not used in the current submission.")

# ============================================================================
# Save preprocessed data
# ============================================================================

preprocessed_data = {
    'data_hierarchical': data_hierarchical,
    'data_regression': data_regression,
    'data_hierarchical_regression': data_hierarchical_regression,
    'data_sarima': data_sarima,  # For SARIMA (optional)
    'data_sarimax': data_sarimax,  # For SARIMAX (optional, not used)
    'day_data': day_data,
    'day_data_sorted': day_data_sorted  # For time series models
}

with open(os.path.join(output_dir, "preprocessed_data.pkl"), 'wb') as f:
    pickle.dump(preprocessed_data, f)

print("\n\nPreprocessing complete! Data saved to output/models/preprocessed_data.pkl")

