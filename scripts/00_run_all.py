# ============================================================================
# Master Script: Run Complete Bayesian Analysis
# Bayesian Bike Sharing Analysis
# ============================================================================
#
# This script runs the complete analysis pipeline:
# 1. Data exploration
# 2. Data preprocessing
# 3. Model fitting (3 Stan models)
# 4. MCMC diagnostics
# 5. Model comparison
# 6. Visualizations
#
# Usage: python scripts/00_run_all.py
# ============================================================================

import time
import subprocess
import sys
import os

def run_script(script_name, step_num, total_steps):
    """Run a Python script and handle errors."""
    print(f"[{step_num}/{total_steps}] Running {script_name}...")
    try:
        # Get the full path to the script
        script_path = os.path.join(script_dir, script_name)
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=script_dir,
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ Complete\n")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {script_name}")
        if e.stderr:
            print(e.stderr)
        if e.stdout:
            print(e.stdout)
        return False

print("=" * 40)
print("BAYESIAN BIKE SHARING ANALYSIS")
print("Complete Pipeline Execution")
print("=" * 40 + "\n")

start_time = time.time()

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Step 1: Data Exploration
if not run_script("01_data_exploration.py", 1, 6):
    sys.exit(1)

# Step 2: Data Preprocessing
if not run_script("02_data_preprocessing.py", 2, 6):
    sys.exit(1)

# Step 3: Fit Models (this may take 10-30 minutes)
print("[3/6] Fitting Bayesian models...")
print("  This may take 10-30 minutes depending on your system.")
print("  Progress will be shown for each model...\n")
# Try simplified version first (no Stan compilation required)
if not run_script("06_fit_models_simple.py", 3, 6):
    # If simplified fails, try full Stan version
    print("Trying full Stan version...")
    if not run_script("06_fit_models.py", 3, 6):
        print("Warning: Model fitting failed. Continuing with available results...")

# Step 4: MCMC Diagnostics
if not run_script("07_mcmc_diagnostics.py", 4, 6):
    sys.exit(1)

# Step 5: Model Comparison
if not run_script("08_model_comparison.py", 5, 6):
    sys.exit(1)

# Step 6: Visualizations
if not run_script("09_visualizations.py", 6, 6):
    sys.exit(1)

# Summary
end_time = time.time()
elapsed = (end_time - start_time) / 60

print("=" * 40)
print("ANALYSIS COMPLETE!")
print("=" * 40)
print(f"Total time: {elapsed:.2f} minutes")
print("\nOutput files:")
print("  - Figures: output/figures/")
print("  - Tables: output/tables/")
print("  - Models: output/models/")
print("\nNext steps:")
print("  1. Review MCMC diagnostics (output/tables/07_mcmc_diagnostics_summary.csv)")
print("  2. Check model comparison (output/tables/08_loo_comparison.csv)")
print("  3. Examine visualizations (output/figures/)")
print("=" * 40)

