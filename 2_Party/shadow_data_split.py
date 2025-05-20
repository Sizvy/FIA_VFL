import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

os.makedirs('shadow_model_data', exist_ok=True)

# Load and shuffle data
df = pd.read_csv('../../datasets/drive_cleaned.csv')
data = df.values
np.random.shuffle(data)

# Shadow model parameters
total_shadow_models = 5
shadow_data_size = 4000
train_ratio = 0.8
shadow_total = total_shadow_models * shadow_data_size

# Split shadow data (first 20,000 records)
shadow_data = data[:shadow_total]
victim_data = data[shadow_total:]

# Save victim data
np.save(f'shadow_model_data/victim_data.npy', victim_data)

# Split and save shadow models
for i in range(1, total_shadow_models + 1):
    start_idx = (i-1) * shadow_data_size
    end_idx = i * shadow_data_size
    model_data = shadow_data[start_idx:end_idx]
    
    train, test = train_test_split(model_data, train_size=train_ratio)
    np.save(f'shadow_model_data/shadow_{i}_train.npy', train)
    np.save(f'shadow_model_data/shadow_{i}_test.npy', test)
