import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

def load_client_data(client_id):
    """Load and normalize data"""
    assert client_id in [1, 2], "Client ID must be 1 or 2"
    
    train_data = np.load(f'splitted_data/client_{client_id}_train.npy')
    val_data = np.load(f'splitted_data/client_{client_id}_val.npy')
    test_data = np.load(f'splitted_data/client_{client_id}_test.npy')
    
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

class BottomModel(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=64, output_dim=64):
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
    def __init__(self, input_dim=128, num_classes=11):
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
                   criterion, optimizer1, optimizer2_bottom, optimizer_top, device):
    """Enhanced training with gradient clipping and proper batch handling"""
    client1_bottom.train()
    client2_bottom.train()
    top_model.train()
    
    running_loss = 0.0
    for (x1, y), (x2,) in zip(client1_loader, client2_loader):
        x1, x2, y = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        batch_size = min(x1.size(0), x2.size(0), y.size(0))  # Take the minimum batch size
        x1 = x1[:batch_size]
        x2 = x2[:batch_size]
        y = y[:batch_size]

        # Forward pass
        optimizer1.zero_grad()
        optimizer2_bottom.zero_grad()
        optimizer_top.zero_grad()
        
        h1 = client1_bottom(x1)
        h2 = client2_bottom(x2)
        h_combined = torch.cat([h1, h2], dim=1)
        outputs = top_model(h_combined)
        
        # Backward pass with gradient clipping
        loss = criterion(outputs, y)
        loss.backward()
        
        # Gradient clipping for all models
        torch.nn.utils.clip_grad_norm_(client1_bottom.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(client2_bottom.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(top_model.parameters(), 1.0)
        
        optimizer1.step()
        optimizer2_bottom.step()
        optimizer_top.step()
        
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50   # Increased epochs
    patience = 5     # More patience
    batch_size = 128  # Larger batch size
    
    # Load and verify data
    client1_data = load_client_data(1)
    client2_data = load_client_data(2)
    
    print(f"Client 1 samples: {len(client1_data[0])}")
    print(f"Client 2 samples: {len(client2_data[0])}")
    
    # Create dataloaders
    train_loader1, val_loader1, test_loader1 = create_dataloaders(
        *client1_data, batch_size=batch_size
    )
    train_loader2, val_loader2, test_loader2 = create_dataloaders(
        *client2_data[:3], batch_size=batch_size
    )

    # Initialize models (unchanged architecture)
    client1_bottom = BottomModel(input_dim=client1_data[0].shape[1], output_dim=64).to(device)
    client2_bottom = BottomModel(input_dim=client2_data[0].shape[1], output_dim=64).to(device)
    top_model = TopModel().to(device)

    # Enhanced optimizers with weight decay
    optimizer1 = optim.AdamW(client1_bottom.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer2_bottom = optim.Adam(client2_bottom.parameters(), lr=0.001)
    optimizer_top = optim.Adam(top_model.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=num_epochs)
    scheduler2_bottom = optim.lr_scheduler.CosineAnnealingLR(optimizer2_bottom, T_max=num_epochs)
    scheduler_top = optim.lr_scheduler.CosineAnnealingLR(optimizer_top, T_max=num_epochs)

    # Label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    counter = 0

    # Training loop with early stopping based on accuracy
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            train_loader1, train_loader2,
            client1_bottom, client2_bottom, top_model,
            criterion, optimizer1, optimizer2_bottom, optimizer_top, device
        )
        
        val_loss, val_acc, val_f1 = validate(
            val_loader1, val_loader2,
            [client1_bottom, client2_bottom, top_model],
            criterion, device
        )
        
        scheduler1.step()
        scheduler2_bottom.step()
        scheduler_top.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(f"Current LR: {optimizer1.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
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

    # Final evaluation
    checkpoint = torch.load('Models/best_vfl_model.pt')
    client1_bottom.load_state_dict(checkpoint['client1_bottom'])
    client2_bottom.load_state_dict(checkpoint['client2_bottom'])
    top_model.load_state_dict(checkpoint['top_model'])
    
    test_loss, test_acc, test_f1 = validate(
        test_loader1, test_loader2,
        [client1_bottom, client2_bottom, top_model],
        criterion, device
    )

    print("\nFinal Results:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()
