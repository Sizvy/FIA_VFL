import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def add_dp_noise(data, epsilon, delta=1e-5):
    # Calculate per-feature sensitivity based on actual data ranges
    data_ranges = data.max(axis=0) - data.min(axis=0)
    sensitivities = data_ranges / len(data)  # Normalized per-sample influence
    
    # Calculate noise scale per feature
    sigma = sensitivities * np.sqrt(2 * np.log(1.25/delta)) / epsilon
    
    # Add adaptive noise
    noise = np.random.normal(0, sigma, size=data.shape)
    noisy_data = data + noise
    
    return noisy_data

def load_client_data(client_id, dp_epsilon=None):
    """Improved data loading with better DP handling"""
    # Load raw data
    train_data = np.load(f'splitted_data/client_{client_id}_train.npy')
    val_data = np.load(f'splitted_data/client_{client_id}_val.npy')
    test_data = np.load(f'splitted_data/client_{client_id}_test.npy')
    
    # Store original stats for validation/test normalization
    original_mean = train_data.mean(axis=0)
    original_std = train_data.std(axis=0)
    
    # Apply DP noise if requested
    if dp_epsilon is not None:
        # Clip data to reduce sensitivity
        train_data = np.clip(train_data, 
                           np.percentile(train_data, 5), 
                           np.percentile(train_data, 95))
        
        # Add smarter DP noise
        train_data = add_dp_noise(train_data, dp_epsilon)
        
        # Verify noise levels
        noise_pct = np.abs(train_data - np.load(f'splitted_data/client_{client_id}_train.npy')).mean() / original_std.mean()
        print(f"Added {noise_pct*100:.2f}% noise relative to feature std")
    
    # Normalize using StandardScaler for better numerical stability
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = (val_data - scaler.mean_) / (scaler.scale_ + 1e-8)
    test_data = (test_data - scaler.mean_) / (scaler.scale_ + 1e-8)
    
    if client_id == 1:
        y_train = np.load('splitted_data/client_1_train_labels.npy')
        y_val = np.load('splitted_data/client_1_val_labels.npy')
        y_test = np.load('splitted_data/client_1_test_labels.npy')
        return train_data, val_data, test_data, y_train, y_val, y_test
    
    return train_data, val_data, test_data

def create_dataloaders(*data, batch_size=64):
    """Create dataloaders with improved numerical stability"""
    # Convert to tensors with consistent float32 precision
    tensors = [torch.as_tensor(d, dtype=torch.float32) for d in data[:3]]
    
    if len(data) > 3:
        datasets = [
            TensorDataset(tensors[0], torch.as_tensor(data[3], dtype=torch.long)),
            TensorDataset(tensors[1], torch.as_tensor(data[4], dtype=torch.long)),
            TensorDataset(tensors[2], torch.as_tensor(data[5], dtype=torch.long))
        ]
    else:
        datasets = [TensorDataset(t) for t in tensors]
    
    loaders = [
        DataLoader(
            d,
            batch_size=batch_size,
            shuffle=(i == 0),
            drop_last=True,
            pin_memory=True,
            num_workers=min(4, os.cpu_count() // 2)  # Safer worker count
        )
        for i, d in enumerate(datasets)
    ]
    return loaders
