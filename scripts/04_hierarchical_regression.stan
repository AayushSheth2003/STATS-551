// ============================================================================
// Stan Model: Hierarchical Regression with Normal Likelihood
// Hierarchical regression by season with predictors (temp, hum, windspeed, etc.)
// ============================================================================

data {
  int<lower=0> N;              // Number of observations
  int<lower=1> K;              // Number of groups (seasons)
  int<lower=1> P;              // Number of predictors
  matrix[N, P] X;              // Design matrix (standardized)
  int<lower=1, upper=K> group[N];  // Group indicator (1-indexed)
  vector[N] y;                 // Response variable (bike rental count)
}

parameters {
  // Hierarchical intercepts (centered parameterization with scale, not precision)
  real mu_alpha;                   // Overall intercept mean
  real<lower=0> sigma_alpha;       // SD of group-specific intercepts
  vector[K] alpha;                 // Group-specific intercepts

  // Hierarchical regression coefficients
  vector[P] mu_beta;               // Overall coefficients mean
  vector<lower=0>[P] sigma_beta;   // SD of group-specific coefficients
  matrix[K, P] beta;               // Group-specific coefficients

  // Residual standard deviation
  real<lower=0> sigma;             // Within-group standard deviation
}

transformed parameters {
  vector[N] mu;                    // Linear predictor
  for (n in 1:N) {
    mu[n] = alpha[group[n]] + X[n] * beta[group[n]]';
  }
}

model {
  // Priors for hierarchical intercepts
  mu_alpha ~ normal(4500, 500);      // Centered on typical daily count
  sigma_alpha ~ normal(0, 1000);     // Half-normal prior (scale on intercepts)
  alpha ~ normal(mu_alpha, sigma_alpha); // Group-specific intercepts

  // Priors for hierarchical coefficients
  // Predictors have been standardized, so coefficients should be O(1)
  mu_beta ~ normal(0, 1);
  sigma_beta ~ normal(0, 1);        // Half-normal prior on coefficient scales
  for (k in 1:K) {
    beta[k] ~ normal(mu_beta, sigma_beta);
  }

  // Prior for residual standard deviation (half-normal)
  sigma ~ normal(0, 1500);

  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  real log_lik[N];
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
  }
}

