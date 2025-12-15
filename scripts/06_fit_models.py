# ============================================================================
# Script 06: Fit Bayesian Models using CmdStanPy (no PyStan)
# Bayesian Bike Sharing Analysis
# ============================================================================

import pickle
import time
import os
import numpy as np
import arviz as az
from cmdstanpy import CmdStanModel, cmdstan_path, install_cmdstan

# Get script directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, "output", "models")
os.makedirs(output_dir, exist_ok=True)

def ensure_cmdstan():
    """
    Ensure CmdStan is installed and available for cmdstanpy.
    This may take several minutes the first time it runs.
    """
    try:
        _ = cmdstan_path()
    except Exception:
        print("CmdStan not found. Installing CmdStan (one-time setup, may take several minutes)...")
        install_cmdstan()


# Load preprocessed data
with open(os.path.join(output_dir, "preprocessed_data.pkl"), 'rb') as f:
    preprocessed_data = pickle.load(f)

data_hierarchical = preprocessed_data['data_hierarchical']
data_hierarchical_regression = preprocessed_data.get('data_hierarchical_regression')

print("=== FITTING BAYESIAN MODELS ===\n")

# ============================================================================
# Model 1: Hierarchical Normal Model
# ============================================================================

print("Fitting Model 1: Hierarchical Normal Model by Season...")

# Ensure CmdStan is available
ensure_cmdstan()

# Compile Stan model via CmdStanPy
print("Compiling Model 1...")
model1 = CmdStanModel(stan_file=os.path.join(script_dir, "03_hierarchical_model.stan"))

# Prepare data for CmdStan
data1 = {
    'N': int(data_hierarchical['N']),
    'K': int(data_hierarchical['K']),
    'y': data_hierarchical['y'].tolist(),
    'group': (data_hierarchical['group'] + 1).tolist()  # 1-indexed groups
}

# Fit model (more conservative sampler settings to improve convergence)
print("Sampling Model 1 (iter_sampling=2000, iter_warmup=2000, adapt_delta=0.99, max_treedepth=15)...")
fit1 = model1.sample(
    data=data1,
    chains=4,
    parallel_chains=4,
    iter_sampling=2000,
    iter_warmup=2000,
    seed=123,
    adapt_delta=0.99,
    max_treedepth=15
)

print("Model 1 complete!")
print("Saving fit...")

# Save fit using arviz
fit1_az = az.from_cmdstanpy(posterior=fit1)
fit1_az.to_netcdf(os.path.join(output_dir, "fit_hierarchical.nc"))

# ============================================================================
# Model 1.5: Hierarchical Regression (Normal Likelihood)
# ============================================================================


# NOTE:
# With the updated priors in `04_hierarchical_regression.stan`, this model
# should sample more stably. We now fit it when data is available.

if data_hierarchical_regression is not None:
    print("\nFitting Model 1.5: Hierarchical Regression (Normal Likelihood)...")
    stan_file = os.path.join(script_dir, "04_hierarchical_regression.stan")
    if os.path.exists(stan_file):
        with open(stan_file, 'r') as f:
            model1_5_code = f.read()
        print("Compiling Model 1.5...")
        model1_5 = CmdStanModel(stan_file=stan_file)
        data1_5 = {
            'N': int(data_hierarchical_regression['N']),
            'K': int(data_hierarchical_regression['K']),
            'P': int(data_hierarchical_regression['P']),
            'X': data_hierarchical_regression['X'].tolist(),
            'group': (data_hierarchical_regression['group']).tolist(),
            'y': data_hierarchical_regression['y'].tolist()
        }
        print("Sampling Model 1.5 (experimental)...")
        fit1_5 = model1_5.sample(
            data=data1_5,
            chains=4,
            parallel_chains=4,
            iter_sampling=2000,
            iter_warmup=2000,
            seed=123,
            adapt_delta=0.99,
            max_treedepth=15
        )
        fit1_5_az = az.from_cmdstanpy(posterior=fit1_5)
        fit1_5_az.to_netcdf(os.path.join(output_dir, "fit_hierarchical_regression.nc"))
        print("Model 1.5 complete (experimental).")

# ============================================================================
# Model 2: Hierarchical Negative Binomial Regression
# ============================================================================

print("\nFitting Model 2: Hierarchical Negative Binomial Regression...")

# Compile Stan model via CmdStanPy
print("Compiling Model 2...")
model2 = CmdStanModel(stan_file=os.path.join(script_dir, "04_hierarchical_negative_binomial.stan"))

# Prepare data for CmdStan (use hierarchical regression data structure)
data2 = {
    'N': int(data_hierarchical_regression['N']),
    'K': int(data_hierarchical_regression['K']),
    'P': int(data_hierarchical_regression['P']),
    'X': data_hierarchical_regression['X'].tolist(),
    'group': (data_hierarchical_regression['group']).tolist(),  # Already 1-indexed
    'y': data_hierarchical_regression['y'].astype(int).tolist()  # Negative Binomial requires integer
}

# Fit model (more robust settings)
print("Sampling Model 2 (iter_sampling=2000, iter_warmup=2000, adapt_delta=0.99, max_treedepth=15)...")
fit2 = model2.sample(
    data=data2,
    chains=4,
    parallel_chains=4,
    iter_sampling=2000,
    iter_warmup=2000,
    seed=123,
    adapt_delta=0.99,
    max_treedepth=15
)

print("Model 2 complete!")
print("Saving fit...")

# Save fit using arviz
fit2_az = az.from_cmdstanpy(posterior=fit2)
fit2_az.to_netcdf(os.path.join(output_dir, "fit_hierarchical_negative_binomial.nc"))

# ============================================================================
# Model 3: SARIMAX (Optional, not used in current submission)
# ============================================================================

# NOTE: SARIMAX model is included as an optional reference but is not fitted
# in the current submission. The Stan file `05_bayesian_sarima.stan` can be
# extended to include exogenous regressors (SARIMAX), but this is not part
# of the active workflow.
#
# If you want to fit SARIMAX (not recommended for this submission):
# 1. Extend `05_bayesian_sarima.stan` to include exogenous regressors X
# 2. Prepare data_sarimax from preprocessed_data (if available)
# 3. Uncomment and modify the code below:
#
# if False:  # Set to True only if you want to fit SARIMAX
#     print("\nFitting Model 3: SARIMAX (Time Series with Exogenous Regressors)...")
#     stan_file = os.path.join(script_dir, "05_bayesian_sarima.stan")
#     if os.path.exists(stan_file):
#         data_sarimax = preprocessed_data.get('data_sarimax')
#         if data_sarimax is not None:
#             print("Compiling Model 3 (SARIMAX)...")
#             model3 = CmdStanModel(stan_file=stan_file)
#             # Prepare data for SARIMAX (would need to be defined)
#             # ... fitting code here ...
#             # fit3_az.to_netcdf(os.path.join(output_dir, "fit_sarimax.nc"))
#             print("Model 3 (SARIMAX) complete.")
#         else:
#             print("Model 3 (SARIMAX): data_sarimax not found, skipping.")

# ============================================================================
# Extract log-likelihoods for model comparison
# ============================================================================

print("\nExtracting log-likelihoods for model comparison...")

# Compute LOO-CV using arviz
# Note: log_lik must be in the Stan model's generated quantities block
try:
    loo1 = az.loo(fit1_az, pointwise=True)
    if fit2_az is not None:
        loo2 = az.loo(fit2_az, pointwise=True)
    else:
        loo2 = None
except Exception as e:
    print(f"Warning: Could not compute LOO-CV. Error: {e}")
    print("This may be because log_lik is not in the Stan model output.")
    print("Continuing without LOO-CV comparison...")
    loo1 = loo2 = None
    comparison = None

if loo1 is not None:
    print("\n=== LOO-CV RESULTS ===")
    print("Model 1 (Hierarchical):")
    print(loo1)
    print("\nModel 2 (Hierarchical Negative Binomial):")
    print(loo2)
    
    # Compare models (only include valid models)
    print("\n=== MODEL COMPARISON ===")
    model_dict = {"Model 1: Hierarchical": loo1,
                  "Model 2: Hierarchical Negative Binomial": loo2}
    comparison = az.compare(model_dict)
    print(comparison)
else:
    comparison = None

# Save comparison results
comparison_results = {
    'loo1': loo1,
    'loo2': loo2,
    'comparison': comparison
}

with open(os.path.join(output_dir, "model_comparison.pkl"), 'wb') as f:
    pickle.dump(comparison_results, f)

print("\n\nAll models fitted! Results saved to output/models/")
print(f"Total computation time: {time.time():.2f} seconds")

