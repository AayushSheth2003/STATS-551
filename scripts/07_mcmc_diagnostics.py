# ============================================================================
# Script 07: MCMC Diagnostics
# Bayesian Bike Sharing Analysis
# ============================================================================

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import os

# Get script directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Create output directories if they don't exist
figures_dir = os.path.join(project_root, "output", "figures")
tables_dir = os.path.join(project_root, "output", "tables")
models_dir = os.path.join(project_root, "output", "models")
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

# Load fitted models (skip if files don't exist)
fit1_az = None
fit2_az = None
fit3_az = None

if os.path.exists(os.path.join(models_dir, "fit_hierarchical.nc")):
    fit1_az = az.from_netcdf(os.path.join(models_dir, "fit_hierarchical.nc"))
if os.path.exists(os.path.join(models_dir, "fit_bayesian_gaussian_process.nc")):
    fit2_az = az.from_netcdf(os.path.join(models_dir, "fit_bayesian_gaussian_process.nc"))
if os.path.exists(os.path.join(models_dir, "fit_bayesian_structural_time_series.nc")):
    fit3_az = az.from_netcdf(os.path.join(models_dir, "fit_bayesian_structural_time_series.nc"))

print("=== MCMC DIAGNOSTICS ===\n")

# ============================================================================
# Model 1: Hierarchical Normal Model
# ============================================================================

print("Diagnostics for Model 1: Hierarchical Normal Model")

# Trace plots
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
az.plot_trace(fit1_az, var_names=['mu', 'tau', 'sigma', 'theta'], 
              axes=axes, compact=True)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "07_model1_trace.png"), dpi=300)
plt.close()

# R-hat statistics
rhat1 = az.rhat(fit1_az)
print("\nR-hat statistics (should be < 1.01):")
print(rhat1)
max_rhat1 = float(rhat1.max().values)
print(f"Max R-hat: {max_rhat1:.4f}")

# Effective sample size
neff1 = az.ess(fit1_az)
print("\nEffective sample size:")
print(neff1)
min_neff1 = float(neff1.min().values)
print(f"Min ESS: {min_neff1:.0f}")

# Posterior distributions
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_forest(fit1_az, var_names=['mu', 'tau', 'sigma', 'theta'], 
               combined=True, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "07_model1_posterior.png"), dpi=300)
plt.close()

# ============================================================================
# Model 2: Hierarchical Negative Binomial Regression
# ============================================================================

print("\n\nDiagnostics for Model 2: Bayesian Gaussian Process Regression")

# Trace plots
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
az.plot_trace(fit2_az, var_names=['alpha', 'sigma', 'length_scale', 'mu'], 
              axes=axes, compact=True)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "07_model2_trace.png"), dpi=300)
plt.close()

# R-hat statistics
rhat2 = az.rhat(fit2_az)
print("\nR-hat statistics:")
print(rhat2)
max_rhat2 = float(rhat2.max().values)
print(f"Max R-hat: {max_rhat2:.4f}")

# Effective sample size
neff2 = az.ess(fit2_az)
print("\nEffective sample size:")
print(neff2)
min_neff2 = float(neff2.min().values)
print(f"Min ESS: {min_neff2:.0f}")

# Posterior distributions
fig, ax = plt.subplots(figsize=(12, 8))
az.plot_forest(fit2_az, var_names=['alpha', 'sigma', 'length_scale', 'mu'], 
               combined=True, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "07_model2_posterior.png"), dpi=300)
plt.close()

# ============================================================================
# Model 3: Bayesian ARIMA (Time Series)
# ============================================================================

if fit3_az is not None:
    print("\n\nDiagnostics for Model 3: Bayesian Structural Time Series")
    
    # Trace plots
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    az.plot_trace(fit3_az, var_names=['sigma_obs', 'sigma_level', 'sigma_slope', 'beta', 'phi_nb'], 
                  axes=axes, compact=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "07_model3_trace.png"), dpi=300)
    plt.close()
    
    # R-hat statistics
    rhat3 = az.rhat(fit3_az)
    print("\nR-hat statistics:")
    print(rhat3)
    max_rhat3 = float(rhat3.max().values)
    print(f"Max R-hat: {max_rhat3:.4f}")
    
    # Effective sample size
    neff3 = az.ess(fit3_az)
    print("\nEffective sample size:")
    print(neff3)
    min_neff3 = float(neff3.min().values)
    print(f"Min ESS: {min_neff3:.0f}")
    
    # Posterior distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_forest(fit3_az, var_names=['sigma_obs', 'sigma_level', 'sigma_slope', 'beta', 'phi_nb'], 
                   combined=True, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "07_model3_posterior.png"), dpi=300)
    plt.close()
else:
    print("\n\nModel 3: Bayesian Structural Time Series - File not found, skipping diagnostics")
    max_rhat3 = None
    min_neff3 = None

# ============================================================================
# Summary diagnostics table
# ============================================================================

# Build diagnostics summary (only for models that exist)
summary_rows = []

if fit1_az is not None:
    n_samples1 = len(fit1_az.posterior.chain) * len(fit1_az.posterior.draw)
    summary_rows.append({
        'Model': 'Hierarchical',
        'Max_Rhat': max_rhat1,
        'Min_ESS': min_neff1,
        'Min_ESS_Ratio': min_neff1/n_samples1 if n_samples1 > 0 else 0,
        'Converged': max_rhat1 < 1.01 and min_neff1/n_samples1 > 0.1 if n_samples1 > 0 else False
    })

if fit2_az is not None:
    n_samples2 = len(fit2_az.posterior.chain) * len(fit2_az.posterior.draw)
    summary_rows.append({
        'Model': 'Bayesian Gaussian Process',
        'Max_Rhat': max_rhat2,
        'Min_ESS': min_neff2,
        'Min_ESS_Ratio': min_neff2/n_samples2 if n_samples2 > 0 else 0,
        'Converged': max_rhat2 < 1.01 and min_neff2/n_samples2 > 0.1 if n_samples2 > 0 else False
    })

if fit3_az is not None and max_rhat3 is not None:
    n_samples3 = len(fit3_az.posterior.chain) * len(fit3_az.posterior.draw)
    summary_rows.append({
        'Model': 'Bayesian Structural Time Series',
        'Max_Rhat': max_rhat3,
        'Min_ESS': min_neff3,
        'Min_ESS_Ratio': min_neff3/n_samples3 if n_samples3 > 0 else 0,
        'Converged': max_rhat3 < 1.01 and min_neff3/n_samples3 > 0.1 if n_samples3 > 0 else False
    })

if summary_rows:
    diagnostics_summary = pd.DataFrame(summary_rows)
else:
    print("No models found for diagnostics!")
    diagnostics_summary = pd.DataFrame()

diagnostics_summary.to_csv(os.path.join(tables_dir, "07_mcmc_diagnostics_summary.csv"), 
                          index=False)

print("\n\n=== DIAGNOSTICS SUMMARY ===")
print(diagnostics_summary)

print("\n\nDiagnostics complete! Figures and tables saved to output/")

