import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--keep_target_feature', action='store_true',
                    help='Keep the target feature in victim data')
args = parser.parse_args()

# Config
os.makedirs('../shadow_model_data', exist_ok=True)
# df = pd.read_csv('../../../datasets/drive_cleaned.csv')
# n = 10
target_feature_idx = -15

# correlations = df.corr()[df.columns[-1]].abs().sort_values(ascending=False)
# top_n_features = correlations[1:n+1].index.tolist()
# print(f"Selected top {n} features based on correlation:\n{top_n_features}")

# filtered_data = df[top_n_features + [df.columns[-1]]].values
# np.random.shuffle(filtered_data)

# shadow_data_size = 20000
# shadow_data = filtered_data[:shadow_data_size]
# victim_data = filtered_data[shadow_data_size:]

victim_data = np.load('../shadow_model_data/victim_data_initial.npy')
print(f"Initial victim data shape: {victim_data.shape}")

if not args.keep_target_feature:
    victim_data = np.delete(victim_data, target_feature_idx, axis=1)
    print(f"\nRemoved target feature")

np.save('../shadow_model_data/victim_data.npy', victim_data)
print(f"\nFinal victim data shape: {victim_data.shape}")
