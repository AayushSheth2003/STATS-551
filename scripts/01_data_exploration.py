# ============================================================================
# Script 01: Data Exploration
# Bayesian Bike Sharing Analysis
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Get script directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, "output", "figures")
os.makedirs(output_dir, exist_ok=True)

# Load data
day_data = pd.read_csv(os.path.join(project_root, "data", "day.csv"))
hour_data = pd.read_csv(os.path.join(project_root, "data", "hour.csv"))

# Convert date
day_data['dteday'] = pd.to_datetime(day_data['dteday'])
hour_data['dteday'] = pd.to_datetime(hour_data['dteday'])

print("=== DATA EXPLORATION ===\n")
print("Daily Data:")
print(day_data.describe())
print("\n\nHourly Data:")
print(hour_data.describe())

# Target variable: cnt (total rental count)
print("\n\n=== TARGET VARIABLE: cnt ===")
print("Daily counts summary:")
print(day_data['cnt'].describe())
print(f"\nMean: {day_data['cnt'].mean():.2f}")
print(f"SD: {day_data['cnt'].std():.2f}")

# Distribution of target variable
plt.figure(figsize=(8, 6))
plt.hist(day_data['cnt'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
plt.title("Distribution of Daily Bike Rental Counts")
plt.xlabel("Count (cnt)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_target_distribution.png"), dpi=300)
plt.close()

# By season
plt.figure(figsize=(8, 6))
season_labels = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
day_data['season_label'] = day_data['season'].map(season_labels)
sns.boxplot(data=day_data, x='season', y='cnt', palette='Set2')
plt.title("Bike Rentals by Season")
plt.xlabel("Season")
plt.ylabel("Count (cnt)")
plt.xticks([0, 1, 2, 3], [season_labels[i] for i in range(1, 5)])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_by_season.png"), dpi=300)
plt.close()

# By weather situation
plt.figure(figsize=(8, 6))
sns.boxplot(data=day_data, x='weathersit', y='cnt', palette='Set2')
plt.title("Bike Rentals by Weather Situation")
plt.xlabel("Weather Situation")
plt.ylabel("Count (cnt)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_by_weather.png"), dpi=300)
plt.close()

# Time series
plt.figure(figsize=(12, 6))
plt.plot(day_data['dteday'], day_data['cnt'], color='steelblue', alpha=0.7, linewidth=1)
# Add LOESS-like smoothing using rolling mean
day_data_sorted = day_data.sort_values('dteday')
rolling_mean = day_data_sorted['cnt'].rolling(window=30, center=True).mean()
plt.plot(day_data_sorted['dteday'], rolling_mean, color='red', linewidth=2, label='Smoothed trend')
plt.title("Daily Bike Rentals Over Time")
plt.xlabel("Date")
plt.ylabel("Count (cnt)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_timeseries.png"), dpi=300)
plt.close()

# Correlation with predictors
numeric_vars = ["temp", "atemp", "hum", "windspeed", "cnt"]
cor_matrix = day_data[numeric_vars].corr()
print("\n\n=== CORRELATION MATRIX ===")
print(cor_matrix.round(3))

# Group statistics by season
print("\n\n=== GROUP STATISTICS BY SEASON ===")
season_stats = day_data.groupby('season')['cnt'].agg([
    ('n', 'count'),
    ('mean_cnt', 'mean'),
    ('sd_cnt', 'std'),
    ('min_cnt', 'min'),
    ('max_cnt', 'max')
]).reset_index()
print(season_stats)

print("\n\nExploration complete! Figures saved to output/figures/")

