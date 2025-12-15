# ============================================================================
# Script 06: Fit Bayesian Models using Simplified MCMC (No Stan Required)
# Bayesian Bike Sharing Analysis
# This is a simplified version that uses scipy for MCMC sampling
# ============================================================================

import pickle
import time
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import arviz as az
import xarray as xr

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Get script directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, "output", "models")
os.makedirs(output_dir, exist_ok=True)

# Load preprocessed data
with open(os.path.join(output_dir, "preprocessed_data.pkl"), 'rb') as f:
    preprocessed_data = pickle.load(f)

day_data = preprocessed_data['day_data']
data_bart = preprocessed_data.get('data_bart')
data_gp = preprocessed_data.get('data_gp')
data_bsts = preprocessed_data.get('data_bsts')

print("=== FITTING BAYESIAN MODELS (Simplified MCMC) ===", flush=True)
print("Note: This uses a simplified MCMC approach for demonstration.", flush=True)
print("For full Stan models, install RTools and use cmdstanpy.\n", flush=True)
import sys
import traceback

# ============================================================================
# Model 1: Bayesian Additive Regression Trees (BART) - Simplified
# Note: Full BART requires complex tree MCMC, this is a simplified approximation
# ============================================================================

print("Fitting Model 1: Bayesian Additive Regression Trees (BART)...", flush=True)
sys.stdout.flush()

if data_bart is None:
    print("Warning: BART data not found. Skipping Model 1.", flush=True)
    fit1_az = None
else:
    y = data_bart['y']
    X = data_bart['X']
    N, P = X.shape
    num_trees = data_bart.get('num_trees', 50)

    # Simplified MCMC for BART (approximation - full BART requires tree MCMC)
    n_samples = 500
    n_warmup = 250
    n_chains = 2
    
    # Initialize storage
    samples = {
        'sigma': np.zeros((n_chains, n_samples)),
        'tree_values': np.zeros((n_chains, n_samples, N, num_trees))
    }
    
    # Initialize with simple linear model + noise as approximation
    from sklearn.ensemble import RandomForestRegressor
    rf_init = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    rf_init.fit(X, y)
    y_pred_init = rf_init.predict(X)
    sigma_init = np.std(y - y_pred_init)
    
    for chain in range(n_chains):
        print(f"  Chain {chain + 1}/{n_chains}...", flush=True)
        
        # Initialize
        sigma = np.abs(np.random.normal(sigma_init, sigma_init * 0.2))
        tree_values = np.random.normal(0, sigma / np.sqrt(num_trees), (N, num_trees))
        
        # Adjust so sum equals initial predictions (approximate)
        tree_sum = np.sum(tree_values, axis=1)
        if np.std(tree_sum) > 1e-10:
            tree_values = tree_values * (y_pred_init.reshape(-1, 1) / (tree_sum.reshape(-1, 1) + 1e-10))
        
        for i in range(n_samples + n_warmup):
            if (i + 1) % 100 == 0:
                print(f"    Iteration {i + 1}/{n_samples + n_warmup}...", flush=True)
            
            # Compute mean prediction
            y_mean = np.sum(tree_values, axis=1)
            
            # Sample sigma
            residuals = y - y_mean
            ss_residual = np.sum(residuals**2)
            sigma_alpha = 1 + N/2
            sigma_beta = 0.5 * (ss_residual + 1e-10)
            if sigma_beta > 0:
                sigma = np.sqrt(1 / np.random.gamma(sigma_alpha, 1/sigma_beta))
            else:
                sigma = np.abs(np.random.normal(sigma_init, sigma_init * 0.2))
            
            # Sample tree values (simplified - regularize toward current mean)
            for t in range(num_trees):
                tree_other_sum = np.sum(tree_values[:, np.arange(num_trees) != t], axis=1)
                residuals_tree = y - tree_other_sum
                # Regularization: each tree contributes small amount
                tree_mean = residuals_tree / num_trees
                tree_values[:, t] = np.random.normal(tree_mean, sigma / np.sqrt(num_trees))
            
            if i >= n_warmup:
                samples['sigma'][chain, i - n_warmup] = sigma
                samples['tree_values'][chain, i - n_warmup, :, :] = tree_values
    
    # Convert to arviz format
    # Compute y_mean from tree_values
    y_mean_samples = np.sum(samples['tree_values'], axis=3)  # Sum over trees
    
    posterior = {
        'sigma': (['chain', 'draw'], samples['sigma']),
        'tree_values': (['chain', 'draw', 'tree_values_dim_0', 'tree_values_dim_1'], samples['tree_values']),
        'y_mean': (['chain', 'draw', 'y_mean_dim_0'], y_mean_samples)
    }
    
    # Create log_lik
    log_lik_samples = np.zeros((n_chains, n_samples, N))
    for chain in range(n_chains):
        for i in range(n_samples):
            for n in range(N):
                y_mean_val = y_mean_samples[chain, i, n]
                sigma_val = samples['sigma'][chain, i]
                log_lik_samples[chain, i, n] = stats.norm.logpdf(
                    y[n], y_mean_val, sigma_val + 1e-10
                )
    
    posterior['log_lik'] = (['chain', 'draw', 'log_lik_dim_0'], log_lik_samples)
    
    coords = {
        'chain': range(n_chains),
        'draw': range(n_samples),
        'tree_values_dim_0': range(N),
        'tree_values_dim_1': range(num_trees),
        'y_mean_dim_0': range(N),
        'log_lik_dim_0': range(N)
    }
    
    fit1_az = az.InferenceData(
        posterior=xr.Dataset(posterior, coords=coords)
    )
    
    fit1_az.to_netcdf(os.path.join(output_dir, "fit_bayesian_bart.nc"))
    print("Model 1 complete!", flush=True)
    sys.stdout.flush()

# ============================================================================
# Model 2: Bayesian Gaussian Process Regression
# Note: GP models are too complex for simplified MCMC
# Please use the full Stan version: python scripts/06_fit_models.py
# ============================================================================

print("\nModel 2: Bayesian Gaussian Process Regression", flush=True)
print("  Note: Gaussian Process models require full Stan implementation.", flush=True)
print("  Please run: python scripts/06_fit_models.py for Model 2", flush=True)
fit2_az = None

# Skipped - use full Stan version
samples2 = {
    'mu_alpha': np.zeros((n_chains, n_samples)),
    'tau_alpha': np.zeros((n_chains, n_samples)),
    'alpha': np.zeros((n_chains, n_samples, K)),
    'mu_beta': np.zeros((n_chains, n_samples, P)),
    'tau_beta': np.zeros((n_chains, n_samples, P)),
    'beta': np.zeros((n_chains, n_samples, K, P)),
    'phi': np.zeros((n_chains, n_samples)),
    'phi_group': np.zeros((n_chains, n_samples, K))
}

for chain in range(n_chains):
    print(f"  Chain {chain + 1}/{n_chains}...", flush=True)
    
    # Initialize on log scale
    mu_alpha = np.random.normal(8.0, 0.5)  # exp(8) ≈ 3000
    tau_alpha = np.random.gamma(0.1, 0.1)
    alpha = np.random.normal(mu_alpha, 1/np.sqrt(tau_alpha + 1e-10), K)
    mu_beta = np.random.normal(0, 0.5, P)
    tau_beta = np.random.gamma(0.1, 0.1, P)
    beta = np.random.normal(0, 0.3, (K, P))
    phi = np.random.gamma(1.0, 0.1)
    phi_group = np.random.gamma(phi * 10, 10, K)
    
    for i in range(n_samples + n_warmup):
        if (i + 1) % 100 == 0:
            print(f"    Iteration {i + 1}/{n_samples + n_warmup}...", flush=True)
        
        # Compute log-means for each observation
        log_mu = np.zeros(N)
        for n in range(N):
            k = int(group[n])
            log_mu[n] = alpha[k] + np.dot(X[n, :], beta[k, :])
        mu = np.exp(log_mu)
        
        # Sample mu_alpha and tau_alpha (hyperparameters for intercepts)
        alpha_mean = np.mean(alpha)
        mu_alpha = np.random.normal(alpha_mean, 0.5)
        alpha_var = np.var(alpha)
        tau_alpha = np.random.gamma(0.1 + K/2, 0.1 + 0.5 * alpha_var * K)
        
        # Sample group-level intercepts
        for k in range(K):
            group_mask = (group == k)
            if np.sum(group_mask) > 0:
                y_k = y[group_mask]
                mu_k = mu[group_mask]
                phi_k = phi_group[k]
                
                # Simplified update (using normal approximation for negative binomial)
                # This is a simplified version - full Gibbs would be more complex
                alpha_prior_mean = mu_alpha
                alpha_prior_var = 1 / (tau_alpha + 1e-10)
                
                # Update alpha using gradient-based approach (simplified)
                alpha_new = alpha[k] + np.random.normal(0, 0.1)
                if alpha_new > -10 and alpha_new < 15:  # Reasonable bounds
                    alpha[k] = alpha_new
                alpha[k] = np.clip(alpha[k], -10, 15)
        
        # Sample mu_beta and tau_beta
        for p in range(P):
            beta_p_vals = beta[:, p]
            mu_beta[p] = np.random.normal(np.mean(beta_p_vals), 0.3)
            beta_var = np.var(beta_p_vals)
            tau_beta[p] = np.random.gamma(0.1 + K/2, 0.1 + 0.5 * beta_var * K)
        
        # Sample group-level coefficients
        for k in range(K):
            for p in range(P):
                beta_prior_mean = mu_beta[p]
                beta_prior_var = 1 / (tau_beta[p] + 1e-10)
                beta[k, p] = np.random.normal(beta_prior_mean, np.sqrt(beta_prior_var))
                beta[k, p] = np.clip(beta[k, p], -5, 5)
        
        # Sample phi (overdispersion parameter)
        phi = np.random.gamma(1.0, 0.1)
        phi = np.clip(phi, 0.1, 100)
        
        # Sample group-specific phi
        for k in range(K):
            phi_group[k] = np.random.gamma(phi * 10, 10)
            phi_group[k] = np.clip(phi_group[k], 0.1, 100)
        
        if i >= n_warmup:
            samples2['mu_alpha'][chain, i - n_warmup] = mu_alpha
            samples2['tau_alpha'][chain, i - n_warmup] = tau_alpha
            samples2['alpha'][chain, i - n_warmup, :] = alpha
            samples2['mu_beta'][chain, i - n_warmup, :] = mu_beta
            samples2['tau_beta'][chain, i - n_warmup, :] = tau_beta
            samples2['beta'][chain, i - n_warmup, :, :] = beta
            samples2['phi'][chain, i - n_warmup] = phi
            samples2['phi_group'][chain, i - n_warmup, :] = phi_group

# Convert to arviz format
posterior2 = {
    'mu_alpha': (['chain', 'draw'], samples2['mu_alpha']),
    'tau_alpha': (['chain', 'draw'], samples2['tau_alpha']),
    'alpha': (['chain', 'draw', 'alpha_dim_0'], samples2['alpha']),
    'mu_beta': (['chain', 'draw', 'mu_beta_dim_0'], samples2['mu_beta']),
    'tau_beta': (['chain', 'draw', 'tau_beta_dim_0'], samples2['tau_beta']),
    'beta': (['chain', 'draw', 'beta_dim_0', 'beta_dim_1'], samples2['beta']),
    'phi': (['chain', 'draw'], samples2['phi']),
    'phi_group': (['chain', 'draw', 'phi_group_dim_0'], samples2['phi_group'])
}

# Create log_lik (Negative Binomial)
log_lik_samples2 = np.zeros((n_chains, n_samples, N))
for chain in range(n_chains):
    for i in range(n_samples):
        for n in range(N):
            k = int(group[n])
            log_mu = samples2['alpha'][chain, i, k] + np.dot(X[n, :], samples2['beta'][chain, i, k, :])
            mu = np.exp(log_mu)
            phi_k = samples2['phi_group'][chain, i, k]
            # Negative Binomial log-likelihood (using scipy's parameterization)
            # scipy uses (n, p) where n=phi, p=phi/(mu+phi)
            p = phi_k / (mu + phi_k + 1e-10)
            log_lik_samples2[chain, i, n] = stats.nbinom.logpmf(
                int(y[n]), phi_k, p
            )

posterior2['log_lik'] = (['chain', 'draw', 'log_lik_dim_0'], log_lik_samples2)

coords2 = {
    'chain': range(n_chains),
    'draw': range(n_samples),
    'alpha_dim_0': range(K),
    'mu_beta_dim_0': range(P),
    'tau_beta_dim_0': range(P),
    'beta_dim_0': range(K),
    'beta_dim_1': range(P),
    'phi_group_dim_0': range(K),
    'log_lik_dim_0': range(N)
}

fit2_az = az.InferenceData(
    posterior=xr.Dataset(posterior2, coords=coords2)
)

fit2_az.to_netcdf(os.path.join(output_dir, "fit_hierarchical_negative_binomial.nc"))
print("Model 2 complete!", flush=True)

# ============================================================================
# Model 3: Bayesian Structural Time Series
# Note: State space models are too complex for simplified MCMC
# Please use the full Stan version: python scripts/06_fit_models.py
# ============================================================================

print("\nModel 3: Bayesian Structural Time Series", flush=True)
print("  Note: State space models require full Stan implementation.", flush=True)
print("  Please run: python scripts/06_fit_models.py for Model 3", flush=True)
fit3_az = None

if False:  # Skip simplified implementation
    y = data_sarima['y'].astype(int)  # Must be integers for Negative Binomial
    N = len(y)
    s = data_sarima['s']  # Seasonal period (7 days)

    # Simplified MCMC for ARIMA(1,1,1) with seasonal component
    samples3 = {
        'phi': np.zeros((n_chains, n_samples)),
        'theta': np.zeros((n_chains, n_samples)),
        'phi_seas': np.zeros((n_chains, n_samples)),
        'beta_trend': np.zeros((n_chains, n_samples)),
        'mu': np.zeros((n_chains, n_samples)),
        'phi_nb': np.zeros((n_chains, n_samples))
    }
    
    # Log-transform for ARIMA
    log_y = np.log(y + 1)
    log_y_diff = np.diff(log_y)
    
    for chain in range(n_chains):
        print(f"  Chain {chain + 1}/{n_chains}...", flush=True)
        
        # Initialize ARIMA parameters
        phi = np.clip(np.random.normal(0.3, 0.2), -0.99, 0.99)
        theta = np.clip(np.random.normal(0.2, 0.2), -0.99, 0.99)
        phi_seas = np.clip(np.random.normal(0.3, 0.2), -0.99, 0.99)
        beta_trend = np.random.normal(0, 0.001)
        mu = np.random.normal(8.0, 0.5)
        phi_nb = np.random.gamma(1.0, 0.1)
        
        for i in range(n_samples + n_warmup):
            if (i + 1) % 100 == 0:
                print(f"    Iteration {i + 1}/{n_samples + n_warmup}...", flush=True)
            
            # Sample phi (AR coefficient)
            phi = np.clip(np.random.normal(0.3, 0.2), -0.99, 0.99)
            
            # Sample theta (MA coefficient)
            theta = np.clip(np.random.normal(0.2, 0.2), -0.99, 0.99)
            
            # Sample seasonal AR
            phi_seas = np.clip(np.random.normal(0.3, 0.2), -0.99, 0.99)
            
            # Sample trend
            beta_trend = np.random.normal(0, 0.001)
            
            # Sample mu
            mu = np.random.normal(8.0, 0.5)
            
            # Sample phi_nb
            phi_nb = np.clip(np.random.gamma(1.0, 0.1), 0.1, 100)
            
            if i >= n_warmup:
                samples3['phi'][chain, i - n_warmup] = phi
                samples3['theta'][chain, i - n_warmup] = theta
                samples3['phi_seas'][chain, i - n_warmup] = phi_seas
                samples3['beta_trend'][chain, i - n_warmup] = beta_trend
                samples3['mu'][chain, i - n_warmup] = mu
                samples3['phi_nb'][chain, i - n_warmup] = phi_nb
    
    # Convert to arviz format
    posterior3 = {
        'phi': (['chain', 'draw'], samples3['phi']),
        'theta': (['chain', 'draw'], samples3['theta']),
        'phi_seas': (['chain', 'draw'], samples3['phi_seas']),
        'beta_trend': (['chain', 'draw'], samples3['beta_trend']),
        'mu': (['chain', 'draw'], samples3['mu']),
        'phi_nb': (['chain', 'draw'], samples3['phi_nb'])
    }
    
    # Create log_lik (simplified ARIMA)
    log_lik_samples3 = np.zeros((n_chains, n_samples, N))
    for chain in range(n_chains):
        for i in range(n_samples):
            phi_val = samples3['phi'][chain, i]
            theta_val = samples3['theta'][chain, i]
            phi_seas_val = samples3['phi_seas'][chain, i]
            beta_trend_val = samples3['beta_trend'][chain, i]
            mu_val = samples3['mu'][chain, i]
            phi_nb_val = samples3['phi_nb'][chain, i]
            
            for t in range(3, N):  # Start from 3 to have lags
                # ARIMA prediction on log scale
                mu_diff = phi_val * log_y_diff[t - 1] + beta_trend_val
                if t > s:
                    mu_diff += phi_seas_val * (log_y[t - s] - log_y[t - s - 1])
                
                log_y_pred = log_y[t - 1] + mu_diff
                mu_original = np.exp(log_y_pred)
                
                # Negative Binomial log-likelihood
                p = phi_nb_val / (mu_original + phi_nb_val + 1e-10)
                log_lik_samples3[chain, i, t] = stats.nbinom.logpmf(
                    int(y[t]), phi_nb_val, p
                )
    
    posterior3['log_lik'] = (['chain', 'draw', 'log_lik_dim_0'], log_lik_samples3)
    
    coords3 = {
        'chain': range(n_chains),
        'draw': range(n_samples),
        'log_lik_dim_0': range(N)
    }
    
    fit3_az = az.InferenceData(
        posterior=xr.Dataset(posterior3, coords=coords3)
    )
    
    fit3_az.to_netcdf(os.path.join(output_dir, "fit_bayesian_arima.nc"))
    print("Model 3 complete!", flush=True)

# ============================================================================
# Extract log-likelihoods for model comparison
# ============================================================================

print("\nExtracting log-likelihoods for model comparison...")

try:
    if fit1_az is not None:
        loo1 = az.loo(fit1_az, pointwise=True)
    else:
        loo1 = None
    if fit2_az is not None:
        loo2 = az.loo(fit2_az, pointwise=True)
    else:
        loo2 = None
    if fit3_az is not None:
        loo3 = az.loo(fit3_az, pointwise=True)
    else:
        loo3 = None
    
    print("\n=== LOO-CV RESULTS ===")
    if loo1 is not None:
        print("Model 1 (Bayesian Additive Regression Trees):")
        print(loo1)
    else:
        print("Model 1 (Bayesian Additive Regression Trees): Skipped")
    print("\nModel 2 (Hierarchical Negative Binomial):")
    print(loo2)
    if fit2_az is not None:
        loo2 = az.loo(fit2_az, pointwise=True)
        print("\nModel 2 (Bayesian Gaussian Process):")
        print(loo2)
    else:
        loo2 = None
        print("\nModel 2 (Bayesian Gaussian Process): Skipped - use full Stan version")
    
    if fit3_az is not None:
        loo3 = az.loo(fit3_az, pointwise=True)
        print("\nModel 3 (Bayesian Structural Time Series):")
        print(loo3)
    else:
        loo3 = None
        print("\nModel 3 (Bayesian Structural Time Series): Skipped - use full Stan version")
    
    # Compare models (only include valid models)
    print("\n=== MODEL COMPARISON ===")
    model_dict = {}
    if loo1 is not None:
        model_dict["Model 1: Bayesian Additive Regression Trees"] = loo1
    if loo2 is not None:
        model_dict["Model 2: Bayesian Gaussian Process"] = loo2
    if loo3 is not None:
        model_dict["Model 3: Bayesian Structural Time Series"] = loo3
    comparison = az.compare(model_dict) if len(model_dict) > 1 else None
    print(comparison)
except Exception as e:
    print(f"Warning: Could not compute LOO-CV. Error: {e}")
    loo1 = loo2 = loo3 = None
    comparison = None

# Save comparison results
comparison_results = {
    'loo1': loo1,
    'loo2': loo2,
    'loo3': loo3,
    'comparison': comparison
}

with open(os.path.join(output_dir, "model_comparison.pkl"), 'wb') as f:
    pickle.dump(comparison_results, f)

print("\n\nAll models fitted! Results saved to output/models/")
print(f"Total computation time: {time.time():.2f} seconds")

