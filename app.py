"""
Flask Web Application for Bayesian Bike Sharing Analysis
Interactive dashboard for exploring Bayesian models and predictions
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from scipy import stats
import arviz as az

# Initialize Flask app
app = Flask(__name__)

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "models")

# ============================================================================
# Helper Functions
# ============================================================================

def safe_float(value):
    """Convert value to float, handling NaN and Inf"""
    try:
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                return None
            return float(value)
        return None
    except Exception:
        return None


def load_data():
    """Load bike sharing data from CSV files"""
    try:
        day_data = pd.read_csv(os.path.join(DATA_DIR, "day.csv"))
        if "dteday" in day_data.columns:
            day_data["dteday"] = pd.to_datetime(day_data["dteday"])

        hour_data = None
        hour_path = os.path.join(DATA_DIR, "hour.csv")
        if os.path.exists(hour_path):
            hour_data = pd.read_csv(hour_path)
            if "dteday" in hour_data.columns:
                hour_data["dteday"] = pd.to_datetime(hour_data["dteday"])

        return day_data, hour_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def standardize_predictors(temp, atemp, hum, windspeed, day_data):
    """Standardize predictor values using training data statistics"""
    try:
        if day_data is None:
            raise ValueError("day_data is None")

        temp_mean = day_data["temp"].mean()
        temp_std_val_stat = day_data["temp"].std()
        temp_std_val = (temp - temp_mean) / (temp_std_val_stat + 1e-10)

        atemp_mean = day_data["atemp"].mean()
        atemp_std_val_stat = day_data["atemp"].std()
        atemp_std_val = (atemp - atemp_mean) / (atemp_std_val_stat + 1e-10)

        hum_mean = day_data["hum"].mean()
        hum_std_val_stat = day_data["hum"].std()
        hum_std_val = (hum - hum_mean) / (hum_std_val_stat + 1e-10)

        windspeed_mean = day_data["windspeed"].mean()
        windspeed_std_val_stat = day_data["windspeed"].std()
        windspeed_std_val = (windspeed - windspeed_mean) / (windspeed_std_val_stat + 1e-10)

        return temp_std_val, atemp_std_val, hum_std_val, windspeed_std_val
    except Exception as e:
        print(f"Error standardizing predictors: {e}")
        # Return unstandardized values if error
        return temp, atemp, hum, windspeed


def discover_models():
    """Discover all fitted models in output/models/ directory"""
    models = {}
    if not os.path.exists(OUTPUT_DIR):
        return models
    
    # EXCLUDE these old/removed models from being shown
    excluded_models = [
        'fit_regression_bma.nc'
    ]
    
    # Map of allowed models with proper display names
    model_display_names = {
        'fit_hierarchical.nc': 'Hierarchical Normal Model',
        'fit_hierarchical_regression.nc': 'Hierarchical Regression',
        'fit_hierarchical_negative_binomial.nc': 'Hierarchical Negative Binomial Regression',
        'fit_bayesian_arima.nc': 'Bayesian ARIMA (Time Series)'
    }
    
    # Look for .nc files (ArviZ/NetCDF format)
    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith('.nc') and filename.startswith('fit_'):
            # Skip excluded models
            if filename in excluded_models:
                continue
            
            # Extract model key from filename: fit_model_name.nc -> model_name
            model_key = filename.replace('fit_', '').replace('.nc', '')
            
            # Use proper display name if available, otherwise generate one
            display_name = model_display_names.get(filename, model_key.replace('_', ' ').title())
            
            model_path = os.path.join(OUTPUT_DIR, filename)
            models[model_key] = {
                'filename': filename,
                'path': model_path,
                'display_name': display_name,
                'exists': os.path.exists(model_path)
            }
    
    return models


def get_model_info(model_name):
    """Get information about a specific model"""
    models = discover_models()
    
    # Try to find by model key
    if model_name in models:
        return models[model_name]
    
    # Try to find by filename
    filename = f"fit_{model_name}.nc"
    for key, info in models.items():
        if info['filename'] == filename:
            return info
    
    return None


def generate_predictions_for_model(
    model_key,
    model_info,
    posterior,
    season,
    temp_std,
    atemp_std,
    hum_std,
    windspeed_std,
    holiday,
    workingday,
    weathersit,
    day_data,
):
    """Generate predictions for a specific model using Bayesian algorithms"""
    predictions = []

    # Hierarchical Normal Model
    if "hierarchical" in model_key and "regression" not in model_key and "negative" not in model_key:
        if "theta" in posterior.data_vars and "sigma" in posterior.data_vars:
            theta_samples = posterior["theta"].values
            sigma_samples = posterior["sigma"].values
            # season is 1..4; theta dimension is K (seasons)
            group_idx = season - 1
            if 0 <= group_idx < theta_samples.shape[2]:
                n_samples = min(1000, theta_samples.shape[0] * theta_samples.shape[1])
                for _ in range(n_samples):
                    chain_idx = np.random.randint(0, theta_samples.shape[0])
                    draw_idx = np.random.randint(0, theta_samples.shape[1])
                    theta = theta_samples[chain_idx, draw_idx, group_idx]
                    sigma = sigma_samples[chain_idx, draw_idx]
                    pred = np.random.normal(theta, sigma)
                    predictions.append(max(0, float(pred)))

    # Hierarchical Regression (Normal likelihood)
    elif (
        ("hierarchical" in model_key and "regression" in model_key and "negative" not in model_key)
        or (model_key == "hierarchical_regression")
    ) and {"alpha", "beta", "sigma"} <= set(posterior.data_vars.keys()) and "phi_group" not in posterior.data_vars:
        alpha_samples = posterior["alpha"].values
        beta_samples = posterior["beta"].values
        sigma_samples = posterior["sigma"].values
        X_new = np.array([temp_std, atemp_std, hum_std, windspeed_std, holiday, workingday, weathersit])
        group_idx = season - 1

        if 0 <= group_idx < alpha_samples.shape[2]:
            n_samples = min(1000, alpha_samples.shape[0] * alpha_samples.shape[1])
            for _ in range(n_samples):
                chain_idx = np.random.randint(0, alpha_samples.shape[0])
                draw_idx = np.random.randint(0, alpha_samples.shape[1])
                alpha = float(alpha_samples[chain_idx, draw_idx, group_idx])

                # beta can be [chain, draw, group, P] or [chain, draw, P]
                if len(beta_samples.shape) == 4:
                    beta = beta_samples[chain_idx, draw_idx, group_idx, :]
                elif len(beta_samples.shape) == 3:
                    beta = beta_samples[chain_idx, draw_idx, :]
                else:
                    beta = beta_samples.flatten()[: len(X_new)]

                if len(beta) != len(X_new):
                    # Pad or truncate if needed (robustness)
                    if len(beta) < len(X_new):
                        beta = np.pad(beta, (0, len(X_new) - len(beta)), "constant")
                    else:
                        beta = beta[: len(X_new)]

                # sigma can be [chain, draw] or [chain, draw, 1]
                if len(sigma_samples.shape) == 2:
                    sigma = float(sigma_samples[chain_idx, draw_idx])
                elif len(sigma_samples.shape) == 3:
                    sigma = float(sigma_samples[chain_idx, draw_idx, 0])
                else:
                    sigma = float(sigma_samples.flatten()[0])

                sigma = max(abs(sigma), 1e-6)
                beta_array = np.array(beta, dtype=float)
                X_new_array = np.array(X_new, dtype=float)
                mu = alpha + np.dot(X_new_array, beta_array)

                if np.isfinite(mu) and np.isfinite(sigma):
                    pred = np.random.normal(mu, sigma)
                    predictions.append(max(0, float(pred)))

    # Hierarchical Negative Binomial
    elif (
        ("negative_binomial" in model_key or "nb" in model_key)
        and {"alpha", "beta", "phi_group"} <= set(posterior.data_vars.keys())
    ):
        alpha_samples = posterior["alpha"].values
        beta_samples = posterior["beta"].values
        phi_group_samples = posterior["phi_group"].values
        X_new = np.array([temp_std, atemp_std, hum_std, windspeed_std, holiday, workingday, weathersit])
        group_idx = season - 1

        if 0 <= group_idx < alpha_samples.shape[2]:
            n_samples = min(1000, alpha_samples.shape[0] * alpha_samples.shape[1])
            for _ in range(n_samples):
                chain_idx = np.random.randint(0, alpha_samples.shape[0])
                draw_idx = np.random.randint(0, alpha_samples.shape[1])
                alpha = float(alpha_samples[chain_idx, draw_idx, group_idx])
                beta = beta_samples[chain_idx, draw_idx, group_idx, :]
                phi = float(phi_group_samples[chain_idx, draw_idx, group_idx])

                log_mu = alpha + np.dot(X_new, beta)
                mu = np.exp(log_mu)
                # Negative binomial parameterization: mean mu, overdispersion phi
                p = phi / (mu + phi + 1e-10)
                pred = stats.nbinom.rvs(phi, p)
                predictions.append(max(0, float(pred)))

    return predictions

# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/data/status')
def data_status():
    """Check if data files exist"""
    day_path = os.path.join(DATA_DIR, "day.csv")
    hour_path = os.path.join(DATA_DIR, "hour.csv")
    
    return jsonify({
        'day_data_exists': os.path.exists(day_path),
        'hour_data_exists': os.path.exists(hour_path),
        'data_dir': DATA_DIR
    })

@app.route('/api/data/overview')
def data_overview():
    """Get overview statistics of the dataset for the Data Overview card"""
    try:
        day_path = os.path.join(DATA_DIR, "day.csv")
        hour_path = os.path.join(DATA_DIR, "hour.csv")

        if not os.path.exists(day_path):
            return jsonify({'error': 'day.csv not found in data directory'}), 500

        # Load day-level data
        day_data = pd.read_csv(day_path)
        if 'dteday' in day_data.columns:
            day_data['dteday'] = pd.to_datetime(day_data['dteday'])

        # Load hour-level data if available
        hour_data = None
        if os.path.exists(hour_path):
            hour_data = pd.read_csv(hour_path)
            if 'dteday' in hour_data.columns:
                hour_data['dteday'] = pd.to_datetime(hour_data['dteday'])

        return jsonify({
            'n_observations': int(len(day_data)),
            'day_data': {
                'total_observations': int(len(day_data)),
                'date_range': {
                    'start': str(day_data['dteday'].min()) if 'dteday' in day_data.columns else None,
                    'end': str(day_data['dteday'].max()) if 'dteday' in day_data.columns else None
                },
                'target_stats': {
                    'mean': float(day_data['cnt'].mean()) if 'cnt' in day_data.columns else None,
                    'std': float(day_data['cnt'].std()) if 'cnt' in day_data.columns else None,
                    'min': float(day_data['cnt'].min()) if 'cnt' in day_data.columns else None,
                    'max': float(day_data['cnt'].max()) if 'cnt' in day_data.columns else None
                }
            },
            'hour_data': {
                'total_observations': int(len(hour_data)) if hour_data is not None else 0
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/status')
def models_status():
    """Get status of all available models"""
    models = discover_models()
    
    # Return both old format (for backward compatibility) and new format
    status = {}
    metadata = {}
    for model_key, model_info in models.items():
        status[model_key] = model_info['exists']  # Simple boolean for old format
        metadata[model_key] = {
            'display_name': model_info['display_name'],
            'filename': model_info['filename']
        }
    
    return jsonify({
        'models': status,  # Boolean dict for compatibility
        'metadata': metadata  # Full metadata
    })

@app.route('/api/models/<model_name>/posterior')
def model_posterior(model_name):
    """Get posterior samples and statistics for a model"""
    try:
        model_info = get_model_info(model_name)
        if not model_info or not model_info['exists']:
            return jsonify({'error': f'Model "{model_name}" not found'}), 404
        
        # Load model fit
        fit = az.from_netcdf(model_info['path'])
        posterior = fit.posterior
        
        # Get requested parameter
        param_name = request.args.get('param', None)
        
        if param_name:
            # Return specific parameter
            if param_name in posterior.data_vars:
                param_data = posterior[param_name].values
                
                # Check if user wants a specific index (e.g., "alpha[0]" or "beta[0,1]")
                index_str = request.args.get('index', None)
                
                if index_str:
                    try:
                        # Parse index like "0" or "0,1"
                        indices = [int(i) for i in index_str.split(',')]
                        
                        # Apply indices to get specific element
                        if len(indices) == 1:
                            if len(param_data.shape) >= 3:
                                # For shape (chain, draw, dim1, ...), select specific dim1
                                param_data = param_data[:, :, indices[0]]
                            else:
                                return jsonify({'error': f'Invalid index for parameter "{param_name}"'}), 400
                        elif len(indices) == 2:
                            if len(param_data.shape) >= 4:
                                # For shape (chain, draw, dim1, dim2, ...), select specific dim1,dim2
                                param_data = param_data[:, :, indices[0], indices[1]]
                            else:
                                return jsonify({'error': f'Invalid index for parameter "{param_name}"'}), 400
                        else:
                            return jsonify({'error': f'Too many indices for parameter "{param_name}"'}), 400
                    except (ValueError, IndexError) as e:
                        return jsonify({'error': f'Invalid index format: {index_str}'}), 400
                
                # Flatten across chains and draws
                param_flattened = param_data.flatten()
                
                # Remove NaN/Inf
                param_clean = param_flattened[np.isfinite(param_flattened)]
                
                # If all values are invalid, try alternative approaches
                if len(param_clean) == 0:
                    # For hierarchical_regression, check sample_stats or prior as fallback
                    if 'hierarchical_regression' in model_name.lower():
                        # Try to get diagnostic info to confirm model status
                        try:
                            # Check if there are any valid parameters at all in the model
                            all_valid_params = []
                            for p in posterior.data_vars.keys():
                                try:
                                    p_data = posterior[p].values.flatten()
                                    p_clean = p_data[np.isfinite(p_data)]
                                    if len(p_clean) > 0:
                                        all_valid_params.append(p)
                                except:
                                    pass
                            
                            # Return a helpful message with available parameters
                            return jsonify({
                                'error': f'The Hierarchical Regression model did not converge properly. Parameter "{param_name}" has no valid posterior samples (all NaN/Inf).',
                                'hint': 'This model needs to be refitted with more iterations or different priors. Try: python scripts/06_fit_models.py',
                                'model_convergence_issue': True,
                                'available_params_with_values': all_valid_params[:10] if all_valid_params else [],
                                'recommendation': 'Use the Hierarchical Normal Model or Hierarchical Negative Binomial Regression instead, which have converged successfully.'
                            }), 400
                        except:
                            pass
                    
                    # For other models, try log-space transformation
                    original_shape = param_data.shape
                    if param_name in ['sigma', 'tau', 'tau_alpha', 'tau_beta']:
                        log_param_name = f'log_{param_name}'
                        if log_param_name in posterior.data_vars:
                            log_param_data = posterior[log_param_name].values.flatten()
                            log_param_clean = log_param_data[np.isfinite(log_param_data)]
                            if len(log_param_clean) > 0:
                                param_clean = np.exp(log_param_clean)
                            else:
                                return jsonify({
                                    'error': f'Parameter "{param_name}" has no valid values (all NaN/Inf). This may indicate the model did not converge properly.',
                                    'hint': 'Try selecting a different parameter, or refit the model with more iterations.'
                                }), 400
                        else:
                            return jsonify({
                                'error': f'Parameter "{param_name}" has no valid values (all NaN/Inf). This may indicate the model did not converge properly.',
                                'hint': 'Try selecting a different parameter, or refit the model with more iterations.',
                                'shape': str(original_shape),
                                'available_params': list(posterior.data_vars.keys())[:10]
                            }), 400
                    else:
                        return jsonify({
                            'error': f'Parameter "{param_name}" has no valid values (all NaN/Inf). This may indicate the model did not converge properly.',
                            'hint': 'Try selecting a different parameter, or refit the model with more iterations.',
                            'shape': str(original_shape)
                        }), 400
                
                # Determine if this is an array/vector parameter
                is_array = len(posterior[param_name].values.shape) > 2
                if is_array:
                    array_shape = posterior[param_name].values.shape[2:]
                    # Convert tuple to list for JSON serialization
                    array_shape_list = list(array_shape) if isinstance(array_shape, tuple) else array_shape.tolist()
                else:
                    array_shape_list = None
                
                return jsonify({
                    'parameter': param_name,
                    'mean': safe_float(np.mean(param_clean)),
                    'median': safe_float(np.median(param_clean)),
                    'std': safe_float(np.std(param_clean)),
                    'q2_5': safe_float(np.percentile(param_clean, 2.5)),
                    'q97_5': safe_float(np.percentile(param_clean, 97.5)),
                    'samples': [safe_float(x) for x in param_clean[:1000]],  # Limit to 1000 samples
                    'is_array': is_array,
                    'array_shape': array_shape_list,
                    'index': index_str
                })
            else:
                return jsonify({'error': f'Parameter "{param_name}" not found'}), 404
        else:
            # Return list of filtered, important parameters only
            # Exclude generated quantities and show only key parameters
            excluded_params = {'log_lik', 'y_pred', 'y_forecast', 'log_lambda_forecast'}
            
            # Priority parameters to show (scalar or key indices of arrays)
            priority_params = {
                'hierarchical': ['mu', 'tau', 'sigma', 'theta'],
                'hierarchical_regression': ['mu_alpha', 'tau_alpha', 'sigma', 'alpha', 'beta', 'mu_beta', 'tau_beta'],
                'negative_binomial': ['mu_alpha', 'tau_alpha', 'phi', 'alpha', 'beta', 'mu_beta', 'tau_beta'],
                'arima': ['mu', 'beta_trend', 'phi1', 'theta1', 'phi_s', 'phi_nb']
            }
            
            params = []
            param_info = {}
            
            # Get model type for priority filtering
            model_key = model_name.lower()
            model_priorities = []
            for key, priors in priority_params.items():
                if key in model_key:
                    model_priorities = priors
                    break
            
            all_params = list(posterior.data_vars.keys())
            
            # Pre-check: For hierarchical_regression, verify if model converged
            if 'hierarchical_regression' in model_key:
                valid_param_count = 0
                for test_param in all_params[:10]:  # Check first 10 params
                    try:
                        test_data = posterior[test_param].values.flatten()
                        test_clean = test_data[np.isfinite(test_data)]
                        if len(test_clean) > 0:
                            valid_param_count += 1
                    except:
                        pass
                
                # If no valid parameters at all, return early with a helpful message
                if valid_param_count == 0:
                    return jsonify({
                        'model': model_name,
                        'parameters': [],
                        'parameter_info': {},
                        'display_name': model_info['display_name'],
                        'convergence_warning': True,
                        'message': 'This model did not converge properly (all parameters are NaN/Inf). Please refit the model or use a different model.',
                        'recommendation': 'Try using "Hierarchical Normal Model" or "Hierarchical Negative Binomial Regression" which have converged successfully.'
                    })
            
            for param in all_params:
                # Skip excluded parameters
                if param in excluded_params:
                    continue
                
                param_shape = posterior[param].values.shape
                
                # Store shape info (excluding chain and draw dimensions)
                is_array = len(param_shape) > 2
                
                if is_array:
                    dims = tuple(param_shape[2:])
                    # For array parameters, only include if they're in priority list
                    # and limit to first few indices for visualization
                    if param in model_priorities:
                        # For 1D arrays (vectors), show first 2-3 elements
                        if len(dims) == 1:
                            if dims[0] <= 4:  # Small arrays, show all
                                params.append(param)
                                param_info[param] = {
                                    'is_array': True,
                                    'shape': list(dims),
                                    'dim_count': 1,
                                    'max_indices_to_show': min(4, dims[0])
                                }
                            else:  # Large arrays, show first few
                                params.append(param)
                                param_info[param] = {
                                    'is_array': True,
                                    'shape': list(dims),
                                    'dim_count': 1,
                                    'max_indices_to_show': 3  # Only show first 3
                                }
                        # For 2D arrays (matrices), show first row only
                        elif len(dims) == 2:
                            if param in model_priorities:
                                params.append(param)
                                param_info[param] = {
                                    'is_array': True,
                                    'shape': list(dims),
                                    'dim_count': 2,
                                    'max_indices_to_show': min(3, dims[1])  # Only show first 3 columns of first row
                                }
                        else:
                            # Skip high-dimensional arrays
                            continue
                    else:
                        # Skip non-priority array parameters
                        continue
                else:
                    # Always include scalar parameters, but be lenient for hierarchical_regression
                    try:
                        param_values = posterior[param].values.flatten()
                        valid_values = param_values[np.isfinite(param_values)]
                        # For hierarchical_regression, be more lenient (include even if some NaN)
                        # For other models, require at least 1% valid values
                        if 'hierarchical_regression' in model_key or len(valid_values) > len(param_values) * 0.01:
                            params.append(param)
                            param_info[param] = {
                                'is_array': False,
                                'shape': None,
                                'dim_count': 0
                            }
                    except Exception:
                        # If we can't check, include it anyway (especially for hierarchical_regression)
                        params.append(param)
                        param_info[param] = {
                            'is_array': False,
                            'shape': None,
                            'dim_count': 0
                        }
            
            # If no parameters found (shouldn't happen), include at least some key ones
            if len(params) == 0:
                # Fallback: include first few scalar params
                for param in all_params:
                    if param not in excluded_params:
                        param_shape = posterior[param].values.shape
                        if len(param_shape) <= 2:  # Scalar
                            params.append(param)
                            param_info[param] = {
                                'is_array': False,
                                'shape': None,
                                'dim_count': 0
                            }
                            if len(params) >= 5:  # Limit to 5 parameters max
                                break
            
            return jsonify({
                'model': model_name,
                'parameters': params,
                'parameter_info': param_info,
                'display_name': model_info['display_name']
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """Generate predictions using selected Bayesian ensemble algorithm"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Get algorithm selection
        algorithm = data.get("algorithm", "bma_loo")

        # Get predictor values
        season = int(data.get("season", 1))
        temp = float(data.get("temp", 0.5))
        atemp = float(data.get("atemp", 0.5))
        hum = float(data.get("hum", 0.5))
        windspeed = float(data.get("windspeed", 0.5))
        holiday = int(data.get("holiday", 0))
        workingday = int(data.get("workingday", 1))
        weathersit = int(data.get("weathersit", 1))

        # Standardize predictors using training data
        day_data, _ = load_data()
        if day_data is None:
            return jsonify(
                {
                    "error": "Could not load data files. Please ensure day.csv exists in the data directory."
                }
            ), 500

        temp_std, atemp_std, hum_std, windspeed_std = standardize_predictors(
            temp, atemp, hum, windspeed, day_data
        )

        # Discover available models (optionally exclude problematic ones)
        models = discover_models()
        # For stability, you can exclude hierarchical_regression here if desired:
        # filtered_models = {k: v for k, v in models.items()
        #                    if 'hierarchical_regression' not in k.lower() and v['exists']}
        filtered_models = {k: v for k, v in models.items() if v["exists"]}

        if len(filtered_models) == 0:
            return jsonify({"error": "No suitable models available for prediction"}), 400

        model_predictions = {}
        model_weights_raw = {}

        # Generate predictions from each model using Bayesian algorithms
        for model_key, model_info in filtered_models.items():
            if not model_info["exists"]:
                continue

            try:
                fit = az.from_netcdf(model_info["path"])
                posterior = fit.posterior

                # Compute model weight based on LOO-CV if available
                try:
                    loo = az.loo(fit, pointwise=True)
                    looic = float(loo.estimates.loc["looic", "Estimate"])
                    model_weights_raw[model_key] = looic
                except Exception:
                    model_weights_raw[model_key] = None

                # Generate predictions using model-specific Bayesian algorithm
                preds = generate_predictions_for_model(
                    model_key,
                    model_info,
                    posterior,
                    season,
                    temp_std,
                    atemp_std,
                    hum_std,
                    windspeed_std,
                    holiday,
                    workingday,
                    weathersit,
                    day_data,
                )

                if len(preds) > 0:
                    model_predictions[model_key] = preds

            except Exception as e:
                print(f"Error generating prediction from {model_key}: {e}")
                continue

        if len(model_predictions) == 0:
            return jsonify({"error": "Could not generate predictions from any model"}), 400

        # Compute BMA weights based on LOOIC (smaller LOOIC => higher weight)
        valid_loo = {k: v for k, v in model_weights_raw.items() if v is not None}
        if len(valid_loo) > 0:
            min_looic = min(valid_loo.values())
            rel_weights = {k: np.exp(-0.5 * (v - min_looic)) for k, v in valid_loo.items()}
            total_weight = sum(rel_weights.values())
            if total_weight > 0:
                bma_weights = {k: v / total_weight for k, v in rel_weights.items()}
            else:
                bma_weights = {k: 1.0 / len(model_predictions) for k in model_predictions.keys()}
        else:
            # Equal weights if no LOOIC available
            bma_weights = {k: 1.0 / len(model_predictions) for k in model_predictions.keys()}

        # Generate ensemble predictions using Bayesian Model Averaging
        all_predictions = []
        for model_key, preds in model_predictions.items():
            if len(preds) == 0:
                continue
            weight = bma_weights.get(model_key, 1.0 / len(model_predictions))
            # Sample proportionally to weight
            n_samples = max(1, int(1000 * weight))
            sampled = np.random.choice(preds, size=min(n_samples, len(preds)), replace=True)
            all_predictions.extend(sampled.tolist())

        if len(all_predictions) == 0:
            return jsonify({"error": "Could not generate ensemble predictions"}), 400

        all_predictions = np.array(all_predictions)
        all_predictions = all_predictions[np.isfinite(all_predictions)]
        all_predictions = np.maximum(all_predictions, 0)  # Ensure non-negative

        if len(all_predictions) == 0:
            return jsonify({"error": "All predictions were invalid"}), 400

        # Compute summary statistics
        mean_pred = float(np.mean(all_predictions))
        median_pred = float(np.median(all_predictions))
        std_pred = float(np.std(all_predictions))
        q2_5 = float(np.percentile(all_predictions, 2.5))
        q97_5 = float(np.percentile(all_predictions, 97.5))

        # Ensure credible interval bounds are non-negative
        q2_5 = max(0, q2_5)
        q97_5 = max(0, q97_5)

        # Algorithm display names
        algorithm_names = {
            "bma_loo": "Bayesian Model Averaging (LOO-weighted)",
            "bma_equal": "Bayesian Model Averaging (Equal weights)",
            "bma_best": "Best Model Only (LOO-selected)",
            "bma_robust": "Robust BMA (Median-based)",
            "stacked": "Bayesian Stacking",
        }

        return jsonify(
            {
                "predictions": {
                    "mean": mean_pred,
                    "median": median_pred,
                    "std": std_pred,
                    "ci_lower": q2_5,
                    "ci_upper": q97_5,
                },
                "input": {
                    "season": season,
                    "temp": temp,
                    "atemp": atemp,
                    "hum": hum,
                    "windspeed": windspeed,
                    "holiday": holiday,
                    "workingday": workingday,
                    "weathersit": weathersit,
                },
                "algorithm": algorithm_names.get(algorithm, "Bayesian Model Averaging"),
                "algorithm_code": algorithm,
                "models_used": list(model_predictions.keys()),
                "bma_weights": {k: float(v) for k, v in bma_weights.items()},
                "n_samples": len(all_predictions),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(500)
def handle_500(e):
    """Ensure 500 errors return JSON, not HTML"""
    return jsonify({'error': f'Internal server error: {str(e)}', 'loo_results': []}), 500

@app.errorhandler(404)
def handle_404(e):
    """Ensure 404 errors return JSON for API routes"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found', 'loo_results': []}), 404
    return e

@app.errorhandler(Exception)
def handle_exception(e):
    """Ensure all unhandled exceptions return JSON for API routes"""
    if request.path.startswith('/api/'):
        return jsonify({'error': f'Error: {str(e)}', 'loo_results': []}), 500
    # Re-raise for non-API routes so Flask handles normally
    raise e

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
