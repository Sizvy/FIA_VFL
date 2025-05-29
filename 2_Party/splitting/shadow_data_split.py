import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Config
os.makedirs('../shadow_model_data', exist_ok=True)
df = pd.read_csv('../../../datasets/drive_cleaned.csv')
n = 10
target_feature_idx = -2

correlations = df.corr()[df.columns[-1]].abs().sort_values(ascending=False)
top_n_features = correlations[1:n+1].index.tolist()
print(f"Selected top {n} features based on correlation:\n{top_n_features}")

filtered_data = df[top_n_features + [df.columns[-1]]].values
np.random.shuffle(filtered_data)

shadow_data_size = 20000
shadow_data = filtered_data[:shadow_data_size]
victim_data = filtered_data[shadow_data_size:]

victim_data_without_F = np.delete(victim_data, target_feature_idx, axis=1)
print(f"\nVictim data shape after removing F: {victim_data_without_F.shape}")

removed_feature = top_n_features[target_feature_idx]
print(f"Removed feature from victim data: '{removed_feature}' (Index {target_feature_idx})")

np.save('../shadow_model_data/victim_data.npy', victim_data_without_F)
np.save('../shadow_model_data/shadow_data.npy', shadow_data)    

print("\nData preparation complete:")
print(f"- Shadow data shape (with F): {shadow_data.shape}")
print(f"- Victim data shape (without F): {victim_data_without_F.shape}")
