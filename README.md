# Bayesian Bike Sharing Analysis – STATS 551 Project

This project analyzes the Washington D.C. bike sharing dataset using Bayesian hierarchical models and exposes the results through a small Flask web application.

The goal is to demonstrate a clear end‑to‑end Bayesian workflow:
- Data exploration
- Hierarchical modeling of bike rental counts
- Model comparison via LOO‑CV (in scripts / Colab)
- Interactive visualisation and prediction through a web dashboard

The repository is intentionally kept simple and focused for submission: only the core models and features that are actually used by the app and the write‑up are kept.

---

## High‑level summary

- **Data**: Daily and hourly bike rental counts from `data/day.csv` and `data/hour.csv`.
- **Main models implemented in this repo**:
  - **Model 1 – Hierarchical Normal model by season** (`03_hierarchical_model.stan`).
  - **Model 1.5 – Hierarchical Regression with Normal likelihood** (`04_hierarchical_regression.stan`, experimental Stan version).
  - **Model 1.5 (Python) – Approximate hierarchical regression** (`scripts/06_fit_hierarchical_regression_python.py`, produces `fit_hierarchical_regression.nc` without Stan).
- **Fitting**:
  - `scripts/06_fit_models.py` uses **CmdStanPy** (CmdStan backend) to fit Stan models where CmdStan is available.
  - `scripts/06_fit_hierarchical_regression_python.py` provides a **Python‑only** approximate hierarchical regression fit, so that `fit_hierarchical_regression.nc` can be generated even without a C++ toolchain.
- **Web app** (`app.py` + `templates/index.html`):
  - Data overview based on `day.csv` and `hour.csv`.
  - Model exploration: posterior plots for any discovered model in `output/models/fit_*.nc`.
  - Prediction section: Bayesian model‑averaging of predictions based on the fitted models.
  - Detailed model comparison and diagnostics are documented separately in a Colab notebook.
- **Model comparison and diagnostics**:
  - Primary script‑based comparison and LOO‑CV in `scripts/08_model_comparison.py`.
  - A dedicated Colab notebook (`collab_project/hierarchical_models_comparison.ipynb`) fits hierarchical models and presents convergence/LOO results in one place.

It is acceptable for the **local working copy** to have only `fit_hierarchical.nc` and `fit_hierarchical_regression.nc` present in `output/models/`; the web app will then show exactly these two models.

---

## Problem statement

**Question.** How can we predict daily bike rental counts in Washington D.C. using Bayesian methods, accounting for seasonal structure and weather effects, while properly quantifying uncertainty?

- **Target variable**: `cnt` (total number of rentals per day).
- **Data**: 731 daily observations (2011–2012), plus supplementary hourly data.
- **Key covariates**:
  - Season indicator (`season`), weather situation (`weathersit`).
  - Normalized temperature (`temp`, `atemp`), humidity (`hum`), wind speed (`windspeed`).
  - Calendar indicators: `holiday`, `workingday`.

The hierarchical structure is primarily at the **season** level (four groups), and regression models use weather and calendar covariates.

---

## Bayesian models in this repository

### Model 1 – Hierarchical Normal model by season

- **File**: `scripts/03_hierarchical_model.stan`
- **Purpose**: Simple hierarchical model to illustrate partial pooling across seasons.
- **Structure**:
  - Level 1: Daily counts `y_n` within each season.
  - Level 2: Season‑specific means `theta_k`.
  - Level 3: Hyperparameters controlling the overall mean and between‑season spread.
- **Likelihood**: `y_n ~ normal(theta[season_n], sigma)`.
- **Key parameters**:
  - Global mean for counts.
  - Season‑specific means.
  - Between‑season scale parameter.
  - Within‑season residual standard deviation.
- **Concepts**: exchangeability, partial pooling, shrinkage, decomposition of within‑ vs between‑season variation.

This is the main model visualised in the web app under the "Hierarchical Normal Model" tab.

### Model 1.5 – Hierarchical regression (Normal likelihood, Stan)

- **File**: `scripts/04_hierarchical_regression.stan`
- **Purpose**: Multilevel regression where both intercepts and slopes vary across seasons.
- **Predictors (per day)**: standardized `temp`, `atemp`, `hum`, `windspeed`, plus `holiday`, `workingday`, `weathersit`.
- **Structure**:
  - Season‑specific intercepts and coefficient vectors.
  - Hierarchical priors with scale parameters (`sigma_alpha`, `sigma_beta`) to improve stability.
- **Likelihood**: `y_n ~ normal(alpha[group_n] + X_n * beta[group_n]', sigma)`.
- **Status**:
  - More complex and sensitive to priors and MCMC settings.
  - Primarily explored in the **Colab notebook**, which is the main source for convergence and comparison results for this Stan version.

### Model 1.5 (Python) – Approximate hierarchical regression

- **File**: `scripts/06_fit_hierarchical_regression_python.py`
- **Purpose**: Provide a second model (`fit_hierarchical_regression.nc`) for the web app using **only Python**, so that no Stan/C++ compilation is needed.
- **Data source**: Uses `data_hierarchical_regression` from `output/models/preprocessed_data.pkl` (produced by `02_data_preprocessing.py`).
- **Method** (approximate):
  - For each season, fit a separate ordinary least squares regression of `cnt` on the standardized predictors.
  - Use the OLS estimate and its classical covariance as an approximate Normal posterior for the regression coefficients.
  - Draw samples for:
    - `beta` with shape `(chain, draw, K, P)` (group‑specific coefficients),
    - `sigma` with shape `(chain, draw)` (global residual scale).
  - Wrap these samples into an ArviZ `InferenceData` object and save as `fit_hierarchical_regression.nc`.
- **Role in the project**:
  - Gives the web app a second model to show under the "Hierarchical Regression" tab.
  - Allows Bayesian‑style visualisation and predictions even if Stan cannot be compiled locally.

### SARIMAX model (optional, not used)

- **File**: `scripts/05_bayesian_sarima.stan` (can be extended to SARIMAX)
- **Purpose**: Time series model with seasonal autoregressive integrated moving average and exogenous regressors.
- **Structure** (conceptual):
  - ARIMA components: autoregressive (AR), differencing (I), moving average (MA).
  - Seasonal components: seasonal AR and MA terms.
  - Exogenous regressors: weather and calendar variables as external inputs.
- **Likelihood**: Typically Normal or Negative Binomial for count data.
- **Status**:
  - This model is **not fitted** by `scripts/06_fit_models.py` in the current submission.
  - It is **not used** in the web app, predictions, or the main LOO‑CV comparison.
  - Included as an optional reference for time series extensions but not part of the active workflow.

---

## Scripts and workflow

### Directory structure (simplified)

```text
.
├── app.py                         # Flask web application
├── run_web_app.py                 # Helper to start the web app
├── requirements.txt               # Python dependencies
├── data/
│   ├── day.csv                    # Daily data (used throughout)
│   └── hour.csv                   # Hourly data (used for overview only)
├── scripts/
│   ├── 00_run_all.py              # Optional pipeline driver
│   ├── 01_data_exploration.py     # Exploratory plots (not required for web app)
│   ├── 02_data_preprocessing.py   # Prepares data and saves preprocessed_data.pkl
│   ├── 03_hierarchical_model.stan # Model 1: hierarchical normal by season
│   ├── 04_hierarchical_regression.stan # Model 1.5: hierarchical regression (Normal, Stan)
│   ├── 06_fit_models.py           # Fits Stan models via CmdStanPy (if available)
│   ├── 06_fit_hierarchical_regression_python.py # Python-only approximate hierarchical regression
│   ├── 06_fit_models_simple.py    # Simplified experimental script (not central)
│   ├── 07_mcmc_diagnostics.py     # Optional diagnostics (script‑level)
│   ├── 08_model_comparison.py     # Script‑level model comparison / LOO
│   └── 09_visualizations.py       # Static figures for the report
├── templates/
│   └── index.html                 # Single‑page dashboard (Bootstrap + Plotly)
└── output/
    ├── models/
    │   ├── preprocessed_data.pkl                # Output of 02_data_preprocessing.py
    │   ├── fit_hierarchical.nc                  # Fit of Model 1 (Stan)
    │   ├── fit_hierarchical_regression.nc       # Fit of Model 1.5 (Python-only approx)
    │   └── model_comparison.pkl                 # Optional comparison object from scripts
    └── figures/                                 # Static PNGs used in the report
```

### Typical analysis pipeline (scripts)

1. **Preprocess data**

   ```bash
   python scripts/02_data_preprocessing.py
   ```

   - Reads `data/day.csv` and standardizes continuous predictors.
   - Constructs season indices and regression design matrices.
   - Saves `output/models/preprocessed_data.pkl`.

2. **Fit Stan models (optional, if CmdStan is available)**

   ```bash
   python scripts/06_fit_models.py
   ```

   - Uses CmdStanPy to compile Stan models and sample from:
     - Model 1 (hierarchical normal),
     - Model 1.5 (hierarchical regression).
   - Saves `.nc` files in `output/models/`.

3. **Fit Python‑only hierarchical regression (no Stan required)**

   ```bash
   python scripts/06_fit_hierarchical_regression_python.py
   ```

   - Uses `preprocessed_data.pkl` to build an approximate hierarchical regression fit.
   - Saves `output/models/fit_hierarchical_regression.nc`, which the web app discovers automatically.

4. **Optional: diagnostics and comparison in Python**

   ```bash
   python scripts/07_mcmc_diagnostics.py
   python scripts/08_model_comparison.py
   python scripts/09_visualizations.py
   ```

5. **Colab notebook**

   - `collab_project/hierarchical_models_comparison.ipynb` contains a self‑contained Colab workflow that:
     - Fits the hierarchical normal and hierarchical regression models using CmdStanPy in a managed environment.
     - Checks convergence using ArviZ (R‑hat, ESS).
     - Performs LOO‑CV‑based model comparison.
   - The web dashboard refers to this notebook for detailed model comparison and diagnostics instead of implementing those inside the UI.

---

## Web application

### Overview

The web app is a lightweight Flask application (`app.py`) with a single HTML template (`templates/index.html`). It offers three main pieces of functionality:

1. **Data overview**
   - Backend endpoints:
     - `GET /api/data/status` – whether `day.csv` and `hour.csv` exist.
     - `GET /api/data/overview` – basic summary for the dashboard (N, date range, mean/std/min/max of `cnt`, and hourly sample size).
   - Frontend shows:
     - Total daily and hourly observations.
     - Mean daily rentals and year range.
     - File status indicators for the two CSV files.

2. **Model exploration**
   - Backend endpoints:
     - `GET /api/models/status` – discovers all `fit_*.nc` files in `output/models`.
     - `GET /api/models/<model_name>/posterior` – lists parameters or returns posterior samples for a chosen parameter.
   - Frontend behaviour:
     - Automatically creates tabs for available models (e.g. `hierarchical`, `hierarchical_regression`), based purely on what `.nc` files exist.
     - Populates a parameter dropdown for each model (shows a small curated subset of parameters).
     - When a parameter is selected, requests posterior samples and draws a Plotly histogram with summary statistics.
   - No full model comparison or diagnostics are computed inside the app; it only visualises posterior summaries.

3. **Predictions (Bayesian model averaging)**
   - Backend endpoint:
     - `POST /api/predict` – generates predictive draws using a simple Bayesian model‑averaging scheme across all available models.
   - Workflow:
     - Uses `load_data()` to compute standardisation statistics from `day.csv`.
     - Standardises the user‑input covariates.
     - For each fitted model (any `.nc` file in `output/models`):
       - Loads the ArviZ `InferenceData`.
       - Tries `arviz.loo(fit, pointwise=True)` to obtain **LOOIC**.
       - Uses `generate_predictions_for_model(...)` to draw posterior predictive samples for the chosen covariates.
     - Computes **LOO‑CV‑based weights**:
       - Smaller LOOIC ⇒ larger weight, via the standard formula
         \(w_i \propto \exp(-\tfrac{1}{2}(\text{LOOIC}_i - \min_j \text{LOOIC}_j))\).
       - If LOOIC is unavailable, falls back to equal weights.
     - Returns summary statistics (mean, median, standard deviation, approximate 95% credible interval) and metadata for the models used.
   - Frontend:
     - Provides a form for season, weather and calendar covariates.
     - Shows summary cards and a distribution plot of the predictive distribution.
   - If only one model (e.g. `fit_hierarchical.nc`) is present, Bayesian model averaging naturally reduces to using that single model.

### Prediction algorithms and formulae

The web app supports multiple Bayesian ensemble prediction algorithms. For each algorithm, let \(K\) denote the number of available models, and let \(p_k(y^* | y)\) be the posterior predictive distribution for model \(k\) given new covariates \(x^*\).

#### 1. Bayesian Model Averaging (LOO-weighted) — `bma_loo`

**Weight computation:**
\[
w_k = \frac{\exp(-\tfrac{1}{2}(\text{LOOIC}_k - \min_j \text{LOOIC}_j))}{\sum_{j=1}^{K} \exp(-\tfrac{1}{2}(\text{LOOIC}_j - \min_j \text{LOOIC}_j))}
\]

where \(\text{LOOIC}_k\) is the Leave-One-Out Information Criterion for model \(k\):
\[
\text{LOOIC}_k = -2 \sum_{i=1}^{n} \log p_k(y_i | y_{-i})
\]

**Ensemble prediction:**
\[
p(y^* | y) = \sum_{k=1}^{K} w_k \cdot p_k(y^* | y)
\]

This is the default algorithm and is currently implemented for all algorithm selections.

#### 2. Bayesian Model Averaging (Equal weights) — `bma_equal`

**Weight computation:**
\[
w_k = \frac{1}{K} \quad \text{for all } k
\]

**Ensemble prediction:**
\[
p(y^* | y) = \frac{1}{K} \sum_{k=1}^{K} p_k(y^* | y)
\]

This gives equal weight to all models regardless of their LOOIC values.

#### 3. Best Model Only (LOO-selected) — `bma_best`

**Weight computation:**
\[
w_k = \begin{cases}
1 & \text{if } k = \arg\min_j \text{LOOIC}_j \\
0 & \text{otherwise}
\end{cases}
\]

**Ensemble prediction:**
\[
p(y^* | y) = p_{k^*}(y^* | y)
\]

where \(k^* = \arg\min_k \text{LOOIC}_k\) is the model with the lowest LOOIC.

#### 4. Robust BMA (Median-based) — `bma_robust`

**Weight computation:**
Uses the same LOOIC-based weights as `bma_loo`, but the final prediction uses the **median** of the combined predictive samples rather than the mean.

**Ensemble prediction:**
\[
\hat{y}^* = \text{median}\left(\bigcup_{k=1}^{K} \{y^*_j \sim p_k(y^* | y) : j = 1, \ldots, n_k\}\right)
\]

where \(n_k\) is the number of samples drawn from model \(k\) (proportional to \(w_k\)).

#### 5. Bayesian Stacking — `stacked`

**Weight computation (theoretical):**
Bayesian Stacking optimizes weights to maximize the sum of log-scores:
\[
\max_{w} \sum_{i=1}^{n} \log\left(\sum_{k=1}^{K} w_k \cdot p_k(y_i | y_{-i})\right)
\]

subject to \(w_k \geq 0\) for all \(k\) and \(\sum_{k=1}^{K} w_k = 1\), where \(p_k(y_i | y_{-i})\) is the LOO predictive density for model \(k\) at observation \(i\).

**Ensemble prediction:**
\[
p(y^* | y) = \sum_{k=1}^{K} w_k^{\text{stacked}} \cdot p_k(y^* | y)
\]

**Note:** In the current implementation, Bayesian Stacking uses the same LOOIC-based weights as `bma_loo` (the optimization step is not yet implemented).

### What the web app does not do

- It does **not** run Stan or fit models on the fly.
- It does **not** implement a dynamic LOO‑CV comparison table in the UI.
- It does **not** expose MCMC diagnostics endpoints; these are handled in scripts and in the Colab notebook.

All detailed model comparison tables, LOO‑CV summaries, and diagnostic plots are documented in the accompanying Colab notebook and static figures in `output/figures/`.

---

## Results summary (used in the report)

### Data exploration

Using `scripts/01_data_exploration.py` and direct analysis of `day.csv`:

- **Dataset**: 731 daily observations (2011–2012).
- **Target variable (`cnt`)**:
  - Mean: about 4500 rentals per day.
  - Standard deviation: about 1900.
  - Range: roughly 20 to 8700 rentals per day.
- **Seasonal pattern**:
  - Lowest rentals in spring, highest in summer and fall.
- **Weather effects**:
  - Clear conditions (weathersit = 1) have substantially higher counts than rainy/snowy days.
- **Correlations** (approximate):
  - `temp` and `atemp`: strongly positively correlated with `cnt` (around 0.6).
  - `windspeed` and `hum`: mildly negatively correlated with `cnt`.

These patterns motivated hierarchical grouping by season and regression on weather covariates.

### Model fitting and comparison (conceptual)

- **Model 1 (hierarchical normal)**:
  - Provides season‑specific average rental levels with shrinkage toward a global mean.
  - Good for illustrating partial pooling and the decomposition of within‑ vs between‑season variability.

- **Model 1.5 (hierarchical regression)**:
  - Extends Model 1 by allowing slopes for temperature, humidity, etc. to vary by season.
  - More expressive but also more complex; convergence and fit quality are assessed primarily in the Colab notebook.

In practice, convergence diagnostics (R‑hat, ESS, trace plots) and LOO‑CV summaries are computed using ArviZ in either scripts or the Colab notebook. The Colab notebook is the single, clean source for these numerical results in the submission.

---

## How to run everything

### 1. Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```

Note: Stan models are fitted via **CmdStanPy**, which in turn requires a working CmdStan installation and a C++ toolchain. On some systems this may require additional setup. The Colab notebook and the Python‑only hierarchical regression script are provided to avoid toolchain issues.

### 2. Preprocess data and fit models (local machine)

```bash
python scripts/02_data_preprocessing.py

# Optional: fit Stan models (if CmdStan is available)
python scripts/06_fit_models.py

# Recommended for this submission: fit Python-only hierarchical regression
python scripts/06_fit_hierarchical_regression_python.py
```

This will:
- Prepare `preprocessed_data.pkl`.
- Fit the hierarchical normal model (Stan, if used) and/or the Python‑only hierarchical regression.
- Save `.nc` files (at least `fit_hierarchical.nc` and `fit_hierarchical_regression.nc`) to `output/models`.

### 3. Launch the web app

```bash
python run_web_app.py
# or
python app.py
```

Then open `http://localhost:5000` in a browser.

You should see:
- A data overview card.
- A model status and exploration section showing one or more models based on available `.nc` files.
- A predictions section using Bayesian model averaging over the available fitted models.

### 4. Use the Colab notebook for convergence and comparison

Open `collab_project/hierarchical_models_comparison.ipynb` in Google Colab and run all cells. It will:
- Load the `day.csv` data.
- Fit the hierarchical normal and hierarchical regression models with CmdStanPy in Colab.
- Produce convergence diagnostics (R‑hat, ESS) and LOO‑CV based model comparison.

This notebook is the main reference for detailed numerical results in the report.

---

## Dependencies

Key Python libraries used:

- **NumPy**, **pandas** – data manipulation and numerical computation.
- **ArviZ**, **xarray** – handling Bayesian model output, diagnostics, and `.nc` files.
- **CmdStanPy** – interface to CmdStan (Stan backend) for model fitting (optional locally).
- **Flask** – web framework for the dashboard.
- **Plotly.js** (via CDN in `index.html`) – interactive visualisations.

Everything required is listed in `requirements.txt`.

---
