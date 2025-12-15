# ============================================================================
# Script 08: Model Comparison
# Bayesian Bike Sharing Analysis
# ============================================================================

import pickle
import pandas as pd
import arviz as az
import os
import sys

# Get script directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Create output directory if it doesn't exist
tables_dir = os.path.join(project_root, "output", "tables")
models_dir = os.path.join(project_root, "output", "models")
os.makedirs(tables_dir, exist_ok=True)

# Load model comparison results (skip if not available)
try:
    with open(os.path.join(models_dir, "model_comparison.pkl"), 'rb') as f:
        comparison_results = pickle.load(f)

    loo1 = comparison_results.get('loo1')
    loo2 = comparison_results.get('loo2')
    loo3 = comparison_results.get('loo3')
    comparison = comparison_results.get('comparison')
    
    if loo1 is None and loo2 is None and loo3 is None:
        print("Warning: No LOO-CV results available. Skipping model comparison.")
        sys.exit(0)
except FileNotFoundError:
    print("Warning: Model comparison file not found. Please run 06_fit_models.py first.")
    sys.exit(0)

print("=== MODEL COMPARISON ===\n")

# ============================================================================
# LOO-CV Comparison
# ============================================================================

print("LOO-CV Comparison:")
print(comparison)

# Extract key metrics (only include available models)
model_names = []
loo_results_list = []

if loo1 is not None:
    model_names.append("Model 1: Hierarchical")
    loo_results_list.append(('loo1', loo1))

if loo2 is not None:
    model_names.append("Model 2: Hierarchical Negative Binomial")
    loo_results_list.append(('loo2', loo2))

if loo3 is not None:
    model_names.append("Model 3: Bayesian ARIMA")
    loo_results_list.append(('loo3', loo3))

# Extract LOOIC values properly from arviz loo objects
loo_data = []
for name, (key, loo_obj) in zip(model_names, loo_results_list):
    loo_data.append({
        'Model': name,
        'LOOIC': float(loo_obj.estimates.loc['looic', 'Estimate']),
        'SE_LOOIC': float(loo_obj.estimates.loc['looic', 'SE']),
        'p_loo': float(loo_obj.estimates.loc['p_loo', 'Estimate'])
    })

loo_results = pd.DataFrame(loo_data)

# Add difference from best model
best_idx = loo_results['LOOIC'].idxmin()
loo_results['LOOIC_diff'] = loo_results['LOOIC'] - loo_results.loc[best_idx, 'LOOIC']
loo_results['SE_diff'] = (loo_results['SE_LOOIC']**2 + 
                          loo_results.loc[best_idx, 'SE_LOOIC']**2)**0.5

print("\n\n=== LOO-CV RESULTS TABLE ===")
print(loo_results.round(2))

loo_results.to_csv(os.path.join(tables_dir, "08_loo_comparison.csv"), index=False)

# ============================================================================
# Parameter Estimates Summary
# ============================================================================

print("\n\n=== PARAMETER ESTIMATES ===\n")

# Model 1: Group means
fit1_az = az.from_netcdf(os.path.join(models_dir, "fit_hierarchical.nc"))
theta_summary = az.summary(fit1_az, var_names=['theta'], 
                           kind='stats', hdi_prob=0.95)
print("Model 1 - Group-level means (theta) by season:")
print(theta_summary)

# Model 2: Bayesian Gaussian Process
gp_path = os.path.join(models_dir, "fit_bayesian_gaussian_process.nc")
if os.path.exists(gp_path):
    fit2_az = az.from_netcdf(gp_path)
    gp_summary = az.summary(fit2_az, var_names=['alpha', 'sigma', 'length_scale'], 
                           kind='stats', hdi_prob=0.95)
    print("\n\nModel 2 - GP hyperparameters:")
    print(gp_summary)

# Model 3: BSTS coefficients (if available)
bsts_path = os.path.join(models_dir, "fit_bayesian_structural_time_series.nc")
if os.path.exists(bsts_path):
    fit3_az = az.from_netcdf(bsts_path)
    bsts_summary = az.summary(fit3_az, var_names=['sigma_obs', 'sigma_level', 'sigma_slope', 'beta'], 
                              kind='stats', hdi_prob=0.95)
    print("\n\nModel 3 - BSTS parameters:")
    print(bsts_summary)

# ============================================================================
# Model Selection Interpretation
# ============================================================================

print("\n\n=== MODEL SELECTION INTERPRETATION ===")
print("Lower LOOIC indicates better predictive performance.")
print(f"Best model: {model_names[best_idx]}")
print(f"LOOIC: {loo_results.loc[best_idx, 'LOOIC']:.2f}")

if abs(loo_results.loc[1, 'LOOIC_diff']) < 2 * loo_results.loc[1, 'SE_diff']:
    print("\nNote: Models are not significantly different (within 2 SE).")

print("\n\nModel comparison complete! Results saved to output/tables/")

