import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = '../shadow_model_data/shadow_data.npy'  
OUTPUT_DIR = '../shadow_model_data'        
TARGET_FEATURE_IDX = 0
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_split_shadow_data():
    data = np.load(DATA_PATH)
    X = data[:, :-1]  
    y = data[:, -1]   
    return X, y

def save_client_data(X, y, prefix, split_type='train'):
    num_clients = 2
    client1_features = 10
    client1_data = X[:, :client1_features]
    client2_data = X[:, client1_features:]
    if prefix == 'shadow_plus_F':
        client2_data, _ = np.split(client2_data, 2)
        client1_data, _ = np.split(client1_data, 2)
        y, _ = np.split(y, 2)
        np.save(f'{OUTPUT_DIR}/{prefix}_client_2_{split_type}.npy', client2_data)
    else:
        _, client2_data = np.split(client2_data, 2)
        _, client1_data = np.split(client1_data, 2)
        _, y = np.split(y, 2)
        np.save(f'{OUTPUT_DIR}/{prefix}_client_2_{split_type}.npy', client2_data)

    np.save(f'{OUTPUT_DIR}/{prefix}_client_1_{split_type}.npy', client1_data)
    np.save(f'{OUTPUT_DIR}/{prefix}_client_1_{split_type}_labels.npy', y)
    print(f"Client1 Shape: {client1_data.shape}, Client2 Shape: {client2_data.shape}")

if __name__ == "__main__":
    X, y = load_and_split_shadow_data()
    print(f"Original data shape: {X.shape}, Target feature index: {TARGET_FEATURE_IDX}")
    
    save_client_data(X, y, prefix='shadow_plus_F')
    save_client_data(X, y, prefix='shadow_minus_F')
    
    print(f"""
    Shadow datasets prepared:
    - D+F (with F): 
      * Client 1: {OUTPUT_DIR}/shadow_plus_F_client_1_train.npy
      * Client 2: {OUTPUT_DIR}/shadow_plus_F_client_2_train.npy
    - D-F (without F): 
      * Client 1: {OUTPUT_DIR}/shadow_minus_F_client_1_train.npy
      * Client 2: {OUTPUT_DIR}/shadow_minus_F_client_2_train.npy
    """)
