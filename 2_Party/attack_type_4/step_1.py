import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = '../shadow_model_data/shadow_data.npy'  
OUTPUT_DIR = '../shadow_model_data'        
TARGET_FEATURE_IDX = -14 
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_split_shadow_data():
    data = np.load(DATA_PATH)
    X = data[:, :-1]  
    y = data[:, -1]   
    return X, y

def create_shadow_datasets(X, y, target_feature_idx):
    # D+F: Keep all features (original data)
    X_plus_F = X.copy()
    
    # D-F: Remove target feature F
    X_minus_F = np.delete(X, target_feature_idx, axis=1)
    
    return X_plus_F, X_minus_F, y

def save_client_data(X, y, prefix, split_type='train'):
    num_clients = 2
    features_per_client = (X.shape[1] + num_clients - 1) // num_clients
    
    for i in range(num_clients):
        start_idx = i * features_per_client
        end_idx = min((i + 1) * features_per_client, X.shape[1])
        client_data = X[:, start_idx:end_idx]
        
        np.save(f'{OUTPUT_DIR}/{prefix}_client_{i+1}_{split_type}.npy', client_data)
    
    np.save(f'{OUTPUT_DIR}/{prefix}_client_1_{split_type}_labels.npy', y)

if __name__ == "__main__":
    X, y = load_and_split_shadow_data()
    print(f"Original data shape: {X.shape}, Target feature index: {TARGET_FEATURE_IDX}")
    
    X_plus_F, X_minus_F, y = create_shadow_datasets(X, y, TARGET_FEATURE_IDX)
    print(f"D+F shape: {X_plus_F.shape}, D-F shape: {X_minus_F.shape}")
    
    save_client_data(X_plus_F, y, prefix='shadow_plus_F')
    save_client_data(X_minus_F, y, prefix='shadow_minus_F')
    
    print(f"""
    Shadow datasets prepared:
    - D+F (with F): 
      * Client 1: {OUTPUT_DIR}/shadow_plus_F_client_1_train.npy
      * Client 2: {OUTPUT_DIR}/shadow_plus_F_client_2_train.npy
    - D-F (without F): 
      * Client 1: {OUTPUT_DIR}/shadow_minus_F_client_1_train.npy
      * Client 2: {OUTPUT_DIR}/shadow_minus_F_client_2_train.npy
    """)
