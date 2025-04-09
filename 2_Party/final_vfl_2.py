import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

class Server(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Server, self).__init__()
        self.aggregation_layer = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, client_outputs):
        aggregated = sum(client_outputs)
        return torch.relu(aggregated)
    
    def aggregate(self, client_outputs):
        return self.forward(client_outputs)

class ActiveClient(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActiveClient, self).__init__()
        self.bottom_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.top_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def compute_bottom(self, x):
        return self.bottom_model(x)
        
    def compute_top(self, x):
        return self.top_model(x)

class PassiveClient(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PassiveClient, self).__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        
    def compute_layer(self, x):
        return self.layer(x)

def train_one_epoch(active_loader, passive_loader, active_client, passive_client, server, 
                   criterion, optimizer, device):
    active_client.train()
    passive_client.train()
    server.train()
    
    running_loss = 0.0
    for (x_active, y), (x_passive,) in zip(active_loader, passive_loader):
        x_active, y = x_active.to(device), y.to(device)
        x_passive = x_passive.to(device)
        
        # Passive client computes first layer
        passive_output = passive_client.compute_layer(x_passive)
        
        # Active client computes first layer
        active_output = active_client.compute_bottom(x_active)
        
        # Server aggregates
        combined = server.aggregate([active_output, passive_output])
        
        # Active client completes forward pass
        outputs = active_client.compute_top(combined)
        
        # Backward pass
        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(active_loader)

def evaluate(active_loader, passive_loader, active_client, passive_client, server, criterion, device):
    active_client.eval()
    passive_client.eval()
    server.eval()
    
    val_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for (x_active, y), (x_passive,) in zip(active_loader, passive_loader):
            x_active, y = x_active.to(device), y.to(device)
            x_passive = x_passive.to(device)
            
            passive_output = passive_client.compute_layer(x_passive)
            active_output = active_client.compute_bottom(x_active)
            combined = server.aggregate([active_output, passive_output])
            outputs = active_client.compute_top(combined)
            
            loss = criterion(outputs, y)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds)
    
    return val_loss / len(active_loader), accuracy, f1, precision, recall, report

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    hidden_dim = 64
    
    # Load your pre-split data
    # Active client data (has labels)
    active_train = np.load('splitted_data/client_1_train.npy')
    active_train_labels = np.load('splitted_data/client_1_train_labels.npy')
    active_val = np.load('splitted_data/client_1_val.npy')
    active_val_labels = np.load('splitted_data/client_1_val_labels.npy')
    active_test = np.load('splitted_data/client_1_test.npy')
    active_test_labels = np.load('splitted_data/client_1_test_labels.npy')
    
    # Passive client data (no labels)
    passive_train = np.load('splitted_data/client_2_train.npy')
    passive_val = np.load('splitted_data/client_2_val.npy')
    passive_test = np.load('splitted_data/client_2_test.npy')
    
    # Create dataloaders
    batch_size = 64
    
    # Training data
    active_train_dataset = TensorDataset(
        torch.tensor(active_train, dtype=torch.float32),
        torch.tensor(active_train_labels, dtype=torch.long)
    )
    passive_train_dataset = TensorDataset(
        torch.tensor(passive_train, dtype=torch.float32)
    )
    
    # Validation data
    active_val_dataset = TensorDataset(
        torch.tensor(active_val, dtype=torch.float32),
        torch.tensor(active_val_labels, dtype=torch.long)
    )
    passive_val_dataset = TensorDataset(
        torch.tensor(passive_val, dtype=torch.float32)
    )
    
    # Test data
    active_test_dataset = TensorDataset(
        torch.tensor(active_test, dtype=torch.float32),
        torch.tensor(active_test_labels, dtype=torch.long)
    )
    passive_test_dataset = TensorDataset(
        torch.tensor(passive_test, dtype=torch.float32)
    )
    
    # Create dataloaders
    active_train_loader = DataLoader(active_train_dataset, batch_size=batch_size, shuffle=True)
    passive_train_loader = DataLoader(passive_train_dataset, batch_size=batch_size, shuffle=False)
    
    active_val_loader = DataLoader(active_val_dataset, batch_size=batch_size, shuffle=False)
    passive_val_loader = DataLoader(passive_val_dataset, batch_size=batch_size, shuffle=False)
    
    active_test_loader = DataLoader(active_test_dataset, batch_size=batch_size, shuffle=False)
    passive_test_loader = DataLoader(passive_test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize models
    num_classes = len(np.unique(active_train_labels))
    active_client = ActiveClient(active_train.shape[1], hidden_dim, num_classes)
    passive_client = PassiveClient(passive_train.shape[1], hidden_dim)
    server = Server(hidden_dim, hidden_dim)
    
    # Move to device
    active_client.to(device)
    passive_client.to(device)
    server.to(device)
    
    # Optimizer
    params = list(active_client.parameters()) + \
             list(passive_client.parameters()) + \
             list(server.parameters())
    
    optimizer = optim.Adam(params, lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            active_train_loader, passive_train_loader,
            active_client, passive_client, server,
            criterion, optimizer, device
        )
        
        val_loss, val_acc, val_f1, val_precision, val_recall, val_report = evaluate(
            active_val_loader, passive_val_loader,
            active_client, passive_client, server,
            criterion, device
        )
        
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        print("\nClassification Report:")
        print(val_report)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'active_client_state_dict': active_client.state_dict(),
                'passive_client_state_dict': passive_client.state_dict(),
                'server_state_dict': server.state_dict(),
            }, 'Models/best_model.pth')
    
    # Test the best model
    print("\nTesting best model...")
    checkpoint = torch.load('Models/best_model.pth')
    active_client.load_state_dict(checkpoint['active_client_state_dict'])
    passive_client.load_state_dict(checkpoint['passive_client_state_dict'])
    server.load_state_dict(checkpoint['server_state_dict'])
    
    test_loss, test_acc, test_f1, test_precision, test_recall, test_report = evaluate(
        active_test_loader, passive_test_loader,
        active_client, passive_client, server,
        criterion, device
    )
    
    print("\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print("\nTest Classification Report:")
    print(test_report)
    
    # Save results
    with open('Results/training_results.txt', 'w') as f:
        f.write(f"Final Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
        f.write("\nTest Classification Report:\n")
        f.write(test_report)

if __name__ == "__main__":
    main()
