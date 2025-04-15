import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from opacus import PrivacyEngine

import os
import numpy as np
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
    """Bottom model with DP noise added to outputs"""
    def __init__(self, input_dim=24, hidden_dim=64, output_dim=32, noise_scale=0.5):
        super().__init__()
        self.noise_scale = noise_scale
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.net(x)
        if self.training:  # Only add noise during training
            # Clip values to bound sensitivity
            x = torch.clamp(x, -2.0, 2.0)
            # Add Gaussian noise
            x = x + torch.randn_like(x) * self.noise_scale
        return x

class TopModel(nn.Module):
    """Top model (unchanged)"""
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
                    criterion, optimizer1, optimizer2, top_optimizer,
                    device):
    client1_bottom.train()
    client2_bottom.train()
    top_model.train()

    running_loss = 0.0
    for (x1, y), (x2,) in zip(client1_loader, client2_loader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        top_optimizer.zero_grad()

        h1 = client1_bottom(x1)
        h2 = client2_bottom(x2)
        h_combined = torch.cat([h1, h2], dim=1)
        outputs = top_model(h_combined)

        loss = criterion(outputs, y)
        loss.backward()

        optimizer1.step()
        optimizer2.step()
        top_optimizer.step()

        running_loss += loss.item()

    return running_loss / len(client1_loader)

def validate(client1_loader, client2_loader, models, criterion, device):
    client1_bottom, client2_bottom, top_model = models
    client1_bottom.eval()
    client2_bottom.eval()
    top_model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for (x1, y), (x2,) in zip(client1_loader, client2_loader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            h1 = client1_bottom(x1)
            h2 = client2_bottom(x2)
            h_combined = torch.cat([h1, h2], dim=1)
            outputs = top_model(h_combined)

            loss = criterion(outputs, y)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(client1_loader), acc, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    patience = 5
    noise_multiplier = 1.0
    max_grad_norm = 1.0
    delta = 1e-5

    # Load data
    client1_train, client1_val, client1_test, y_train, y_val, y_test = load_client_data(1)
    client2_train, client2_val, client2_test = load_client_data(2)

    train_loader1, val_loader1, test_loader1 = create_dataloaders(client1_train, client1_val, client1_test, y_train, y_val, y_test)
    train_loader2, val_loader2, test_loader2 = create_dataloaders(client2_train, client2_val, client2_test)

    # Models
    client1_bottom = BottomModel(input_dim=client1_train.shape[1]).to(device)
    client2_bottom = BottomModel(input_dim=client2_train.shape[1]).to(device)
    top_model = TopModel().to(device)

    optimizer1 = optim.Adam(client1_bottom.parameters(), lr=0.001)
    optimizer2 = optim.Adam(client2_bottom.parameters(), lr=0.001)
    top_optimizer = optim.Adam(top_model.parameters(), lr=0.001)

    # Opacus Privacy Engines
    privacy_engine1 = PrivacyEngine()
    client1_bottom, optimizer1, train_loader1 = privacy_engine1.make_private_with_epsilon(
        module=client1_bottom,
        optimizer=optimizer1,
        data_loader=train_loader1,
        target_epsilon=8.0,
        target_delta=delta,
        epochs=num_epochs,
        max_grad_norm=max_grad_norm,
    )

    privacy_engine2 = PrivacyEngine()
    client2_bottom, optimizer2, train_loader2 = privacy_engine2.make_private_with_epsilon(
        module=client2_bottom,
        optimizer=optimizer2,
        data_loader=train_loader2,
        target_epsilon=8.0,
        target_delta=delta,
        epochs=num_epochs,
        max_grad_norm=max_grad_norm,
    )

    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            train_loader1, train_loader2,
            client1_bottom, client2_bottom, top_model,
            criterion,
            optimizer1, optimizer2, top_optimizer,
            device
        )

        val_loss, val_acc, val_f1 = validate(
            val_loader1, val_loader2,
            [client1_bottom, client2_bottom, top_model],
            criterion, device
        )

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

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

    # Report privacy budget
    epsilon1 = privacy_engine1.get_epsilon(delta)
    epsilon2 = privacy_engine2.get_epsilon(delta)
    print(f"Client 1 Privacy ε = {epsilon1:.2f}, δ = {delta}")
    print(f"Client 2 Privacy ε = {epsilon2:.2f}, δ = {delta}")

    # Final test
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

