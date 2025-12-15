"""
Script to compute LOO-CV for models that were already fitted
Use this if the models were fitted before log_lik was added, or if LOO wasn't computed
"""

import arviz as az
import pickle
import os

# Get script directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
models_dir = os.path.join(project_root, "output", "models")

print("Computing LOO-CV for fitted models...\n")

model_files = {
    'loo1': 'fit_hierarchical.nc',
    'loo2': 'fit_bayesian_gaussian_process.nc',
    'loo3': 'fit_bayesian_structural_time_series.nc'
}

loo_results = {}
comparison = None

for loo_key, model_file in model_files.items():
    file_path = os.path.join(models_dir, model_file)
    if not os.path.exists(file_path):
        print(f"Skipping {model_file} - file not found")
        continue
    
    try:
        print(f"Loading {model_file}...")
        fit = az.from_netcdf(file_path)
        
        print(f"Computing LOO-CV for {model_file}...")
        loo_obj = az.loo(fit, pointwise=True)
        loo_results[loo_key] = loo_obj
        
        print(f"✓ Success: LOOIC = {loo_obj.estimates.loc['looic', 'Estimate']:.2f}")
        
    except Exception as e:
        print(f"✗ Error for {model_file}: {e}")
        loo_results[loo_key] = None

# Create comparison if we have at least 2 valid LOO objects
valid_loos = {k: v for k, v in loo_results.items() if v is not None}
if len(valid_loos) >= 2:
    print("\nCreating model comparison...")
    model_names = {
        'loo1': "Model 1: Hierarchical",
        'loo2': "Model 2: Bayesian Gaussian Process",
        'loo3': "Model 3: Bayesian Structural Time Series"
    }
    comparison_dict = {model_names[k]: v for k, v in valid_loos.items()}
    comparison = az.compare(comparison_dict)
    print(comparison)

# Save results
comparison_results = {
    'loo1': loo_results.get('loo1'),
    'loo2': loo_results.get('loo2'),
    'loo3': loo_results.get('loo3'),
    'comparison': comparison
}

output_file = os.path.join(models_dir, "model_comparison.pkl")
with open(output_file, 'wb') as f:
    pickle.dump(comparison_results, f)

print(f"\n✓ Results saved to {output_file}")
print("\nDone! You can now view the comparison in the web interface.")

