// ============================================================================
// Stan Model 1: Hierarchical Normal Model by Season
// Groups bike rentals by season with hierarchical structure
// ============================================================================

data {
  int<lower=0> N;              // Number of observations
  int<lower=1> K;              // Number of groups (seasons)
  vector[N] y;                 // Response variable (bike rental count)
  int<lower=1, upper=K> group[N];  // Group indicator (1-indexed)
}

parameters {
  real mu;                     // Overall mean (hyperparameter)
  real<lower=0> tau;           // Between-group precision (controls between-season variability)
  vector[K] theta;             // Group-level means
  real<lower=0> sigma;         // Within-group standard deviation
}

model {
  // Hyperpriors (tighter, more stable)
  // Daily counts are typically around 4500 with sd ~ 2000
  mu ~ normal(4500, 500);          // Center near empirical mean with moderate spread

  // Prior on between-season precision (tau)
  // Gamma(2, 0.5) has mean 4 and sd ~ 2.8 -> avoids extreme values
  tau ~ gamma(2, 0.5);

  // Group-level means with partial pooling around mu
  theta ~ normal(mu, 1.0 / sqrt(tau));

  // Prior for within-season standard deviation (half-normal)
  sigma ~ normal(0, 1500);        // Truncated at 0 by <lower=0>

  // Likelihood
  for (n in 1:N) {
    y[n] ~ normal(theta[group[n]], sigma);
  }
}

generated quantities {
  real y_pred[N];
  real log_lik[N];
  
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | theta[group[n]], sigma);
    y_pred[n] = normal_rng(theta[group[n]], sigma);
  }
}

