import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class EnhancedAttackModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('attack_model_data/confusion_matrix.png')
    plt.close()

def train_attack_model():
    # Load attack data
    all_data = np.load('attack_model_data/training_outputs.npy')
    # attack_test_data = np.load('attack_model_data/attack_test_data.npy')
    # all_data = np.concatenate([attack_train_data, attack_test_data])
    
    print(f"Total records for attack model: {len(all_data):,}")
    print(f" - 'In' samples: {np.sum(all_data[:,-1] == 1):,}")
    print(f" - 'Out' samples: {np.sum(all_data[:,-1] == 0):,}")
    
    # Prepare features: prediction vectors + original class labels
    X = np.column_stack([
        all_data[:, :11],  # Prediction vectors
        all_data[:, 11]    # Original class labels
    ])
    y = all_data[:, -1]    # Membership labels
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Class balancing
    class_counts = np.bincount(y_train.astype(int))
    weights = 1. / class_counts[y_train.astype(int)]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedAttackModel(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    
    # Training loop
    best_auc = 0
    for epoch in range(100):
        model.train()
        train_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Evaluation
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                
                all_probs.extend(outputs.cpu().numpy())
                all_preds.extend((outputs > 0.5).float().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        cm = confusion_matrix(all_labels, all_preds)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Test Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        scheduler.step(auc)
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'Saved_Models/best_attack_model.pt')
            plot_confusion_matrix(cm, classes=['Out', 'In'])
            print("Saved new best model")
    
    print(f"\nBest AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train_attack_model()
