import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from opacus.accountants import GaussianAccountant

def add_dp_noise(data, epsilon, delta=1e-5, sensitivity=1.0):
    # Calculate noise scale (σ) based on privacy budget
    sigma = sensitivity * np.sqrt(2 * np.log(1.25/delta)) / epsilon
    
    # Add Gaussian noise
    noisy_data = data + np.random.normal(0, sigma, size=data.shape)
    
    # Clip to maintain original data bounds
    data_min, data_max = data.min(), data.max()
    return np.clip(noisy_data, data_min, data_max)

def load_client_data(client_id, dp_epsilon=None):
    """Load and normalize data with optional DP noise"""
    
    train_data = np.load(f'splitted_data/client_{client_id}_train.npy')
    val_data = np.load(f'splitted_data/client_{client_id}_val.npy')
    test_data = np.load(f'splitted_data/client_{client_id}_test.npy')
    
    # Add DP noise to training data if requested
    if dp_epsilon is not None:
        train_data = add_dp_noise(train_data, dp_epsilon)
    
    # Normalize features
    mean, std = train_data.mean(axis=0), train_data.std(axis=0)
    train_data = (train_data - mean) / (std + 1e-8)
    val_data = (val_data - mean) / (std + 1e-8)
    test_data = (test_data - mean) / (std + 1e-8)
    
    if client_id == 1:
        y_train = np.load('splitted_data/client_1_train_labels.npy')
        y_val = np.load('splitted_data/client_1_val_labels.npy')
        y_test = np.load('splitted_data/client_1_test_labels.npy')
        return train_data, val_data, test_data, y_train, y_val, y_test
    return train_data, val_data, test_data

def create_dataloaders(*data, batch_size=64):
    """Create balanced dataloaders with pinned memory"""
    datasets = [TensorDataset(torch.tensor(d, dtype=torch.float32)) for d in data[:3]]
    if len(data) > 3:
        datasets[0] = TensorDataset(torch.tensor(data[0], dtype=torch.float32), 
                                  torch.tensor(data[3], dtype=torch.long))
        datasets[1] = TensorDataset(torch.tensor(data[1], dtype=torch.float32),
                                  torch.tensor(data[4], dtype=torch.long))
        datasets[2] = TensorDataset(torch.tensor(data[2], dtype=torch.float32),
                                  torch.tensor(data[5], dtype=torch.long))
    
    loaders = [DataLoader(d, batch_size=batch_size, shuffle=(i==0),
                         drop_last=True, pin_memory=True, num_workers=4)
               for i, d in enumerate(datasets)]
    return loaders
