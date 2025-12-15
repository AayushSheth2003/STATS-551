"""\
Fit a simple hierarchical-style regression model (by season) in pure Python
and save the result as an ArviZ .nc file for use by the web app.

This script does NOT use Stan / CmdStan. Instead, it:
- Loads preprocessed regression data from output/models/preprocessed_data.pkl
- For each season, fits an ordinary least squares regression
- Approximates the posterior of the coefficients as Normal around the OLS
  estimates with covariance based on the classical formula
- Draws samples from these approximate posteriors to build an ArviZ
  InferenceData object
- Saves the result to output/models/fit_hierarchical_regression.nc

This gives the web app a second model to display ("hierarchical_regression")
without requiring a C++ toolchain.
"""

import os
import pickle

import numpy as np
import arviz as az


# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "models")
PREPROC_PATH = os.path.join(OUTPUT_DIR, "preprocessed_data.pkl")
OUT_NC_PATH = os.path.join(OUTPUT_DIR, "fit_hierarchical_regression.nc")


def load_preprocessed_data():
    if not os.path.exists(PREPROC_PATH):
        raise FileNotFoundError(
            f"Could not find preprocessed_data.pkl at {PREPROC_PATH}.\n"
            "Please run scripts/02_data_preprocessing.py first."
        )
    with open(PREPROC_PATH, "rb") as f:
        preproc = pickle.load(f)
    if "data_hierarchical_regression" not in preproc:
        raise KeyError(
            "data_hierarchical_regression not found in preprocessed data.\n"
            "Please make sure scripts/02_data_preprocessing.py ran successfully."
        )
    return preproc["data_hierarchical_regression"]


def fit_per_season_ols(X, y, group, K):
    """Fit separate OLS regressions for each season (group).

    Returns
    -------
    beta_hat : array, shape (K, P)
        OLS estimates for each group.
    Sigma_beta : array, shape (K, P, P)
        Estimated covariance matrices for each group's coefficients.
    sigma2_hat : float
        Pooled residual variance across all groups (for stability).
    """
    N, P = X.shape
    beta_hat = np.zeros((K, P))
    Sigma_beta = np.zeros((K, P, P))

    # Compute pooled residual variance
    residual_ss_total = 0.0
    dof_total = 0

    for k in range(1, K + 1):
        mask = (group == k)
        X_k = X[mask]
        y_k = y[mask]
        if X_k.shape[0] <= P:
            # Not enough data for full regression; fall back to simple mean-only
            beta_hat[k - 1, :] = 0.0
            Sigma_beta[k - 1] = np.eye(P) * 1e6
            continue

        # OLS: beta = (X'X)^{-1} X'y
        XtX = X_k.T @ X_k
        Xty = X_k.T @ y_k

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)

        b_hat_k = XtX_inv @ Xty
        beta_hat[k - 1, :] = b_hat_k

        # Residuals and variance
        y_hat_k = X_k @ b_hat_k
        resid_k = y_k - y_hat_k
        dof_k = max(len(y_k) - P, 1)
        ss_k = float((resid_k ** 2).sum())

        residual_ss_total += ss_k
        dof_total += dof_k

        # Store covariance up to a scalar sigma2 (which we estimate pooled)
        Sigma_beta[k - 1] = XtX_inv

    sigma2_hat = residual_ss_total / max(dof_total, 1)

    # Scale each group's covariance by pooled sigma^2
    for k in range(K):
        Sigma_beta[k] = Sigma_beta[k] * sigma2_hat

    return beta_hat, Sigma_beta, sigma2_hat


def draw_posterior_samples(beta_hat, Sigma_beta, sigma2_hat, n_chains=4, n_draws=1000):
    """Draw approximate posterior samples for beta and sigma.

    We treat each group's coefficients as approximately Normal
    N(beta_hat_k, Sigma_beta_k) and use a simple log-normal style
    variation around sigma^2 for uncertainty.
    """
    K, P = beta_hat.shape

    # Shapes expected by ArviZ: (chain, draw, ...)
    beta_samples = np.zeros((n_chains, n_draws, K, P))
    sigma_samples = np.zeros((n_chains, n_draws))

    # Simple uncertainty model for sigma: log-normal around sigma2_hat**0.5
    sigma_mean = np.sqrt(sigma2_hat)
    log_sigma_sd = 0.25  # weak uncertainty on log-scale

    for c in range(n_chains):
        for d in range(n_draws):
            # Sigma per draw (global)
            log_sigma = np.log(max(sigma_mean, 1e-6)) + np.random.normal(0.0, log_sigma_sd)
            sigma = float(np.exp(log_sigma))
            sigma_samples[c, d] = sigma

            # Coefficients per group
            for k in range(K):
                mean_k = beta_hat[k]
                cov_k = Sigma_beta[k]
                try:
                    beta_k = np.random.multivariate_normal(mean_k, cov_k)
                except np.linalg.LinAlgError:
                    # Fallback to diagonal covariance if numerical issues arise
                    diag_cov = np.diag(np.clip(np.diag(cov_k), 1e-6, None))
                    beta_k = np.random.multivariate_normal(mean_k, diag_cov)
                beta_samples[c, d, k, :] = beta_k

    return beta_samples, sigma_samples


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_preprocessed_data()
    N = int(data["N"])
    K = int(data["K"])
    P = int(data["P"])
    X = np.asarray(data["X"], dtype=float).reshape(N, P)
    y = np.asarray(data["y"], dtype=float).reshape(N)
    group = np.asarray(data["group"], dtype=int).reshape(N)

    # Fit separate OLS by season and build approximate posteriors
    beta_hat, Sigma_beta, sigma2_hat = fit_per_season_ols(X, y, group, K)

    # Draw samples
    n_chains = 4
    n_draws = 1000
    beta_samples, sigma_samples = draw_posterior_samples(
        beta_hat, Sigma_beta, sigma2_hat, n_chains=n_chains, n_draws=n_draws
    )

    # Build ArviZ InferenceData
    posterior = {
        "beta": beta_samples,   # shape: (chain, draw, K, P)
        "sigma": sigma_samples  # shape: (chain, draw)
    }

    idata = az.from_dict(posterior=posterior)

    # Save to NetCDF so the web app can discover it
    idata.to_netcdf(OUT_NC_PATH)
    print(f"Saved approximate hierarchical regression fit to: {OUT_NC_PATH}")


if __name__ == "__main__":
    main()
