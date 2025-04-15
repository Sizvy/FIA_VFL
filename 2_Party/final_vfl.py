import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

def load_client_data(client_id):
    """Load features and labels for a specific client"""
    assert client_id in [1, 2], "Client ID must be 1 or 2"
    
    train_data = np.load(f'splitted_data/client_{client_id}_train.npy')
    val_data = np.load(f'splitted_data/client_{client_id}_val.npy')
    test_data = np.load(f'splitted_data/client_{client_id}_test.npy')
    
    if client_id == 1:
        y_train = np.load('splitted_data/client_1_train_labels.npy')
        y_val = np.load('splitted_data/client_1_val_labels.npy')
        y_test = np.load('splitted_data/client_1_test_labels.npy')
        return train_data, val_data, test_data, y_train, y_val, y_test
    return train_data, val_data, test_data

def create_dataloaders(*data, batch_size=64):
    """Create PyTorch dataloaders from numpy arrays"""
    datasets = [TensorDataset(torch.tensor(d, dtype=torch.float32)) for d in data[:3]]
    if len(data) > 3:  # Client 1 with labels
        datasets[0] = TensorDataset(torch.tensor(data[0], dtype=torch.float32), 
                                  torch.tensor(data[3], dtype=torch.long))
        datasets[1] = TensorDataset(torch.tensor(data[1], dtype=torch.float32),
                                  torch.tensor(data[4], dtype=torch.long))
        datasets[2] = TensorDataset(torch.tensor(data[2], dtype=torch.float32),
                                  torch.tensor(data[5], dtype=torch.long))
    
    loaders = [DataLoader(d, batch_size=batch_size, shuffle=(i==0)) 
               for i, d in enumerate(datasets)]
    return loaders


class BottomModel(nn.Module):
    """Bottom model for each client (same architecture)"""
    def __init__(self, input_dim=24, hidden_dim=64, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)

class TopModel(nn.Module):
    """Top model (owned by active client)"""
    def __init__(self, input_dim=64, num_classes=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

def train_one_epoch(client1_loader, client2_loader, 
                   client1_bottom, client2_bottom, top_model, 
                   criterion, optimizer, device):
    """One training epoch for VFL (now updates BOTH clients)"""
    client1_bottom.train()
    client2_bottom.train()  # Client 2 now trains
    top_model.train()
    
    running_loss = 0.0
    for (x1, y), (x2,) in zip(client1_loader, client2_loader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        
        # Forward pass (WITH gradients for both clients)
        h1 = client1_bottom(x1)
        h2 = client2_bottom(x2) 
        h_combined = torch.cat([h1, h2], dim=1)
        outputs = top_model(h_combined)
        
        # Backward pass (updates ALL models)
        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()  # Updates Client 1, Client 2, and TopModel
        
        running_loss += loss.item()
    
    return running_loss / len(client1_loader)

def validate(client1_loader, client2_loader, models, criterion, device):
    client1_bottom, client2_bottom, top_model = models
    client1_bottom.eval()
    client2_bottom.eval()
    top_model.eval()
    
    val_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for (x1, y), (x2,) in zip(client1_loader, client2_loader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            
            h1 = client1_bottom(x1)
            h2 = client2_bottom(x2)
            outputs = top_model(torch.cat([h1, h2], dim=1))
            
            loss = criterion(outputs, y)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return val_loss / len(client1_loader), accuracy, f1


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    patience = 12
    
    # Load data
    client1_train, client1_val, client1_test, y_train, y_val, y_test = load_client_data(1)
    client2_train, client2_val, client2_test = load_client_data(2)
    
    # Create dataloaders
    train_loader1, val_loader1, test_loader1 = create_dataloaders(
        client1_train, client1_val, client1_test, y_train, y_val, y_test
    )
    train_loader2, val_loader2, test_loader2 = create_dataloaders(
        client2_train, client2_val, client2_test
    )
    
    # Initialize models
    client1_bottom = BottomModel(input_dim=client1_train.shape[1]).to(device)
    client2_bottom = BottomModel(input_dim=client2_train.shape[1]).to(device)
    top_model = TopModel().to(device)
    
    # Optimizer now includes ALL models (Client 1, Client 2, TopModel)
    optimizer = optim.Adam(
        list(client1_bottom.parameters()) + 
        list(client2_bottom.parameters()) + 
        list(top_model.parameters()), 
        lr=0.001, weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training
    best_val_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        # Train (now updates both clients)
        train_loss = train_one_epoch(
            train_loader1, train_loader2,
            client1_bottom, client2_bottom, top_model,
            criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_f1 = validate(
            val_loader1, val_loader2,
            [client1_bottom, client2_bottom, top_model],
            criterion, device
        )
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save({
                'client1_bottom': client1_bottom.state_dict(),
                'client2_bottom': client2_bottom.state_dict(),
                'top_model': top_model.state_dict()
            }, 'Models/best_vfl_model.pt')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break
    
    # Load best model and test
    checkpoint = torch.load('Models/best_vfl_model.pt')
    client1_bottom.load_state_dict(checkpoint['client1_bottom'])
    client2_bottom.load_state_dict(checkpoint['client2_bottom'])
    top_model.load_state_dict(checkpoint['top_model'])
    
    test_loss, test_acc, test_f1 = validate(
        test_loader1, test_loader2,
        [client1_bottom, client2_bottom, top_model],
        criterion, device
    )
    
    print("\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()
