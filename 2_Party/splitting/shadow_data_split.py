import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

os.makedirs('../shadow_model_data', exist_ok=True)

df = pd.read_csv('../../../datasets/drive_cleaned.csv')

correlations = df.corr()[df.columns[-1]].abs().sort_values(ascending=False)

n = 10
top_n_features = correlations[1:n+1].index.tolist()
print(f"Selected top {n} features based on correlation:")
print(top_n_features)

filtered_data = df[top_n_features + [df.columns[-1]]].values
np.random.shuffle(filtered_data)

total_shadow_models = 5
shadow_data_size = 4000
train_ratio = 0.8
shadow_total = total_shadow_models * shadow_data_size

shadow_data = filtered_data[:shadow_total]
victim_data = filtered_data[shadow_total:]

np.save(f'../shadow_model_data/victim_data.npy', victim_data)

# Split and save shadow models
for i in range(1, total_shadow_models + 1):
    start_idx = (i-1) * shadow_data_size
    end_idx = i * shadow_data_size
    model_data = shadow_data[start_idx:end_idx]
    
    train, test = train_test_split(model_data, train_size=train_ratio)
    np.save(f'../shadow_model_data/shadow_{i}_train.npy', train)
    np.save(f'../shadow_model_data/shadow_{i}_test.npy', test)

print("\nData preparation complete:")
print(f"- Total shadow models: {total_shadow_models}")
print(f"- Records per shadow model: {shadow_data_size} (Train: {int(shadow_data_size*train_ratio)}, Test: {shadow_data_size-int(shadow_data_size*train_ratio)})")
print(f"- Features used: {len(top_n_features)} most correlated features")
