import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

def load_client1_data():
    """
    Loads only Client 1's data from .npy files.
    Returns training, validation, and test sets along with their labels.
    """
    # Load Client 1's data
    X_train = np.load('splitted_data/client_1_train.npy')
    X_val = np.load('splitted_data/client_1_val.npy')
    X_test = np.load('splitted_data/client_1_test.npy')
    
    # Load labels (from client 1)
    y_train = np.load('splitted_data/client_1_train_labels.npy')
    y_val = np.load('splitted_data/client_1_val_labels.npy')
    y_test = np.load('splitted_data/client_1_test_labels.npy')
    
    # Verify labels are in range 0-10
    assert set(np.unique(y_train)).issubset(set(range(11))), "Labels must be 0-10"
    assert set(np.unique(y_val)).issubset(set(range(11))), "Labels must be 0-10"
    assert set(np.unique(y_test)).issubset(set(range(11))), "Labels must be 0-10"
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=64):
    """
    Converts numpy arrays to PyTorch tensors and creates DataLoader objects.
    Returns train, validation, and test dataloaders.
    """
    # Convert to PyTorch tensors with proper types
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)  # CrossEntropy expects long integers
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

class Client1Model(nn.Module):
    """Neural network model for Client 1's 24 features with 11-class output"""
    def __init__(self, input_dim=24, hidden_dim=64, num_classes=11):
        super(Client1Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation here - CrossEntropyLoss includes softmax
        return x

def train_model(model, train_loader, val_loader, num_epochs=50, patience=5):
    """
    Trains the model with early stopping based on validation loss.
    Returns the trained model and training history.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion)
        
        # Store metrics
        history['train_loss'].append(train_loss/len(train_loader))
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {history["train_loss"][-1]:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'Models/client1_model.pt')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break
    
    return model, history

def evaluate_model(model, dataloader, criterion=None):
    """
    Evaluates model performance on given dataloader.
    Returns loss (if criterion provided), accuracy and F1 score (macro averaged).
    """
    model.eval()
    preds = []
    true = []
    loss_total = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            if criterion:
                loss_total += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.tolist())
            true.extend(labels.tolist())
    
    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average='macro')  # Macro average for multi-class
    
    if criterion:
        return loss_total/len(dataloader), acc, f1
    return acc, f1

def save_results(metrics, filename='Results/client1_results.csv'):
    """Saves evaluation metrics to a CSV file"""
    pd.DataFrame.from_dict(metrics, orient='index').to_csv(filename, header=['Value'])

def main():
    # Ensure directories exist
    os.makedirs('Models', exist_ok=True)
    os.makedirs('Results', exist_ok=True)
    
    # 1. Load only Client 1's data
    X_train, X_val, X_test, y_train, y_val, y_test = load_client1_data()
    
    # 2. Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # 3. Initialize model with Client 1's feature dimension
    model = Client1Model(input_dim=X_train.shape[1], num_classes=11)
    
    # 4. Train model
    trained_model, history = train_model(model, train_loader, val_loader)
    
    # 5. Load best model and evaluate on test set
    trained_model.load_state_dict(torch.load('Models/client1_model.pt'))
    test_acc, test_f1 = evaluate_model(trained_model, test_loader)
    
    print("\nFinal Test Results (Client 1 Only):")
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1 Score (macro): {test_f1:.4f}')
    
    # 6. Save results
    results = {
        'final_test_accuracy': test_acc,
        'final_test_f1_macro': test_f1,
        'best_val_accuracy': max(history['val_acc']),
        'best_val_f1_macro': max(history['val_f1'])
    }
    save_results(results)

if __name__ == "__main__":
    main()
