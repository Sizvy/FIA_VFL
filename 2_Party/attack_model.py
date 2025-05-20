import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class AttackModel(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=128, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim//2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def train_attack_model():
    # Load attack data
    attack_train_data = np.load('attack_model_data/attack_train_data.npy')
    attack_test_data = np.load('attack_model_data/attack_test_data.npy')
    all_data = np.concatenate([attack_train_data, attack_test_data])
    
    print(f"Total records for attack model: {len(all_data):,}")
    print(f" - 'In' samples: {np.sum(all_data[:,-1] == 1):,}")
    print(f" - 'Out' samples: {np.sum(all_data[:,-1] == 0):,}")
    
    # Split data (80% train, 20% test)
    X = torch.FloatTensor(all_data[:, :-1])
    y = torch.FloatTensor(all_data[:, -1])
    dataset = TensorDataset(X, y)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttackModel(input_dim=X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    
    # Training loop
    best_auc = 0
    for epoch in range(50):
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
        train_loss /= len(train_loader)
        
        scheduler.step(auc)
        
        print(f"Epoch {epoch+1}:")
        print(f"Train Loss: {train_loss:.4f} | Test Acc: {acc:.4f}")
        print(f"Test F1: {f1:.4f} | Test AUC: {auc:.4f}")
        
        # Save best model
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'attack_model_data/best_attack_model.pt')
            print("Saved new best model")
    
    print(f"\nFinal Best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train_attack_model()
