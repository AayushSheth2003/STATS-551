# ============================================================================
# Script 09: Visualizations and Posterior Analysis
# Bayesian Bike Sharing Analysis
# ============================================================================

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import os

# Get script directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Create output directory if it doesn't exist
figures_dir = os.path.join(project_root, "output", "figures")
models_dir = os.path.join(project_root, "output", "models")
os.makedirs(figures_dir, exist_ok=True)

# Load data and models
with open(os.path.join(models_dir, "preprocessed_data.pkl"), 'rb') as f:
    preprocessed_data = pickle.load(f)

day_data = preprocessed_data['day_data']
data_hierarchical = preprocessed_data['data_hierarchical']

fit1_az = az.from_netcdf(os.path.join(models_dir, "fit_hierarchical.nc"))
# Load Model 2 if available
gp_path = os.path.join(models_dir, "fit_bayesian_gaussian_process.nc")
if os.path.exists(gp_path):
    fit2_az = az.from_netcdf(gp_path)
else:
    fit2_az = None
# Load Model 3 if available
bsts_path = os.path.join(models_dir, "fit_bayesian_structural_time_series.nc")
if os.path.exists(bsts_path):
    fit3_az = az.from_netcdf(bsts_path)
else:
    fit3_az = None

print("=== CREATING VISUALIZATIONS ===\n")

# ============================================================================
# Model 1: Hierarchical Model - Group Comparisons
# ============================================================================

print("Creating visualizations for Model 1...")

# Extract posterior samples
post1 = fit1_az.posterior

# Group means comparison
theta_samples = post1['theta'].values.reshape(-1, 4)
theta_df = pd.DataFrame(theta_samples, columns=[f'Season_{i+1}' for i in range(4)])
theta_long = theta_df.melt(var_name='Season', value_name='Mean_Count')

plt.figure(figsize=(10, 6))
sns.violinplot(data=theta_long, x='Season', y='Mean_Count', palette='Set2')
sns.boxplot(data=theta_long, x='Season', y='Mean_Count', width=0.1, 
            boxprops={'alpha': 0.5}, ax=plt.gca())
plt.title("Posterior Distribution of Group Means by Season\nModel 1: Hierarchical Normal Model")
plt.ylabel("Mean Bike Rental Count")
plt.xlabel("Season")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "09_model1_group_means.png"), dpi=300)
plt.close()

# Shrinkage visualization
theta_means = theta_df.mean().values
mu_mean = float(post1['mu'].values.mean())

shrinkage_data = pd.DataFrame({
    'Season': [f'Season {i+1}' for i in range(4)],
    'Group_Mean': theta_means,
    'Overall_Mean': mu_mean
})

plt.figure(figsize=(8, 6))
plt.scatter(shrinkage_data['Season'], shrinkage_data['Group_Mean'], 
           s=100, color='steelblue', zorder=3)
plt.axhline(y=mu_mean, linestyle='--', color='red', linewidth=2, 
           label='Overall Mean')
for i, row in shrinkage_data.iterrows():
    plt.arrow(i, mu_mean, 0, row['Group_Mean'] - mu_mean, 
             head_width=0.1, head_length=50, fc='gray', ec='gray', alpha=0.5)
plt.title("Shrinkage Toward Overall Mean\nRed line: overall mean, Blue points: group means")
plt.ylabel("Mean Count")
plt.xlabel("Season")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "09_model1_shrinkage.png"), dpi=300)
plt.close()

# ============================================================================
# Model 2: Bayesian Gaussian Process - GP Hyperparameters
# ============================================================================

if fit2_az is not None:
    print("Creating visualizations for Model 2...")
    
    post2 = fit2_az.posterior
    
    # GP hyperparameters
    alpha_samples = post2['alpha'].values.flatten()
    sigma_samples = post2['sigma'].values.flatten()
    
    # Length scales (ARD parameters)
    length_scale_samples = post2['length_scale'].values
    predictor_names = ['temp', 'atemp', 'hum', 'windspeed', 
                       'holiday', 'workingday', 'weathersit']
    
    # Create visualization for length scales
    if len(length_scale_samples.shape) == 3:
        # Reshape: (chain, draw, P) -> (all_samples, P)
        ls_reshaped = length_scale_samples.reshape(-1, length_scale_samples.shape[2])
        ls_df = pd.DataFrame(ls_reshaped, columns=predictor_names)
        ls_long = ls_df.melt(var_name='Predictor', value_name='Length_Scale')
        
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=ls_long, x='Predictor', y='Length_Scale', palette='Set2')
        plt.title("Posterior Distribution of Length Scales (ARD)\nModel 2: Bayesian Gaussian Process")
        plt.ylabel("Length Scale")
        plt.xlabel("Predictor")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "09_model2_length_scales.png"), dpi=300)
        plt.close()
    
    # Signal and noise parameters
    gp_params = pd.DataFrame({
        'alpha (Signal)': alpha_samples,
        'sigma (Noise)': sigma_samples
    })
    gp_params_long = gp_params.melt(var_name='Parameter', value_name='Value')
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=gp_params_long, x='Parameter', y='Value', palette='Set2')
    plt.title("Posterior Distribution of GP Hyperparameters\nModel 2: Bayesian Gaussian Process")
    plt.ylabel("Value")
    plt.xlabel("Parameter")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "09_model2_gp_hyperparameters.png"), dpi=300)
    plt.close()
else:
    print("Model 2: Bayesian Gaussian Process - File not found, skipping visualizations")

# ============================================================================
# Model 3: Bayesian Structural Time Series - Components
# ============================================================================

if fit3_az is not None:
    print("Creating visualizations for Model 3...")
    
    post3 = fit3_az.posterior
    
    # Variance parameters
    sigma_obs_samples = post3.get('sigma_obs', None)
    sigma_level_samples = post3.get('sigma_level', None)
    sigma_slope_samples = post3.get('sigma_slope', None)
    
    if sigma_obs_samples is not None and sigma_level_samples is not None:
        variance_params = pd.DataFrame({
            'sigma_obs (Observation)': sigma_obs_samples.values.flatten(),
            'sigma_level (Trend Level)': sigma_level_samples.values.flatten(),
            'sigma_slope (Trend Slope)': sigma_slope_samples.values.flatten() if sigma_slope_samples is not None else np.array([0] * len(sigma_obs_samples.values.flatten()))
        })
        variance_params_long = variance_params.melt(var_name='Parameter', value_name='Value')
        
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=variance_params_long, x='Parameter', y='Value', palette='Set2')
        plt.title("Posterior Distribution of Variance Parameters\nModel 3: Bayesian Structural Time Series")
        plt.ylabel("Standard Deviation")
        plt.xlabel("Parameter")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "09_model3_variance_params.png"), dpi=300)
        plt.close()
    
    # Regression coefficients
    if 'beta' in post3.data_vars:
        beta_samples = post3['beta'].values
        predictor_names = ['temp', 'atemp', 'hum', 'windspeed', 
                           'holiday', 'workingday', 'weathersit']
        if len(beta_samples.shape) == 3:
            beta_reshaped = beta_samples.reshape(-1, beta_samples.shape[2])
            beta_df = pd.DataFrame(beta_reshaped, columns=predictor_names)
            beta_long = beta_df.melt(var_name='Predictor', value_name='Coefficient')
            
            plt.figure(figsize=(10, 6))
            sns.violinplot(data=beta_long, x='Predictor', y='Coefficient', palette='Set2')
            plt.axhline(y=0, linestyle='--', color='red', linewidth=2)
            plt.title("Posterior Distribution of Regression Coefficients\nModel 3: Bayesian Structural Time Series")
            plt.ylabel("Coefficient")
            plt.xlabel("Predictor")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "09_model3_regression_coefs.png"), dpi=300)
            plt.close()
    
    # Overdispersion parameter
    if 'phi_nb' in post3.data_vars:
        phi_nb_samples = post3['phi_nb'].values.flatten()
        plt.figure(figsize=(8, 6))
        plt.hist(phi_nb_samples, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        plt.title("Posterior Distribution of Overdispersion Parameter\nModel 3: Bayesian Structural Time Series")
        plt.xlabel("Phi (Inverse Dispersion)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "09_model3_overdispersion.png"), dpi=300)
        plt.close()
else:
    print("Model 3: Bayesian Structural Time Series - File not found, skipping visualizations")

# ============================================================================
# Posterior Predictive Checks
# ============================================================================

print("Creating posterior predictive checks...")

# Model 1
y_pred1 = post1['y_pred'].values.flatten()
y_obs = data_hierarchical['y']

# Sample a few predictions
np.random.seed(123)
pred_samples1 = np.random.choice(y_pred1, size=min(100*len(y_obs), len(y_pred1)), 
                                replace=False)

plt.figure(figsize=(10, 6))
plt.hist(y_obs, bins=50, color='steelblue', alpha=0.5, 
        label='Observed', edgecolor='black')
plt.hist(pred_samples1, bins=50, color='red', alpha=0.3, 
        label='Predicted', edgecolor='black')
plt.title("Posterior Predictive Check - Model 1\nBlue: observed, Red: predicted")
plt.xlabel("Bike Rental Count")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "09_ppc_model1.png"), dpi=300)
plt.close()

print("\n\nVisualizations complete! All figures saved to output/figures/")

