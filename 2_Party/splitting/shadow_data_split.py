import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

os.makedirs('../shadow_model_data', exist_ok=True)
df = pd.read_csv('../../../datasets/drive_cleaned.csv')

df.columns = df.columns.astype(str)

shadow_data_sz = 20000
n = 23
target_col = df.columns[-1]
correlations = df.corr()[target_col].abs().sort_values(ascending=False)

top_n_features = [
    f for f in correlations.index[1:n+1]  
    if not f.endswith(('.1', '.2'))][:n]

print(f"Original top {n} features (most correlated first):")
print(top_n_features)
# top_n_features_reversed = top_n_features[::-1]
# print(f"\nReversed top {n} features (least correlated first):")
# print(top_n_features_reversed)

filtered_data = df[top_n_features + [target_col]].values
np.random.shuffle(filtered_data)

shadow_data = filtered_data[:shadow_data_sz]
victim_data = filtered_data[shadow_data_sz:]

np.save('../shadow_model_data/shadow_data.npy', shadow_data)
np.save('../shadow_model_data/victim_data_initial.npy', victim_data)

print("\nFinal data shapes:")
print(f"- Shadow data: {shadow_data.shape}")
print(f"- Victim data: {victim_data.shape}")
