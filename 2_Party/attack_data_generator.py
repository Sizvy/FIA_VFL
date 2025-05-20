import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

os.makedirs('attack_model_data', exist_ok=True)

class ShadowBottomModel(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=64, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class ShadowTopModel(nn.Module):
    def __init__(self, input_dim=128, num_classes=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_shadow_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    batch_size = 128
    
    attack_train_data = []
    attack_test_data = []
    
    for i in range(1, 6):
        train_data = np.load(f'shadow_model_data/shadow_{i}_train.npy')
        test_data = np.load(f'shadow_model_data/shadow_{i}_test.npy')
        
        X_train = torch.FloatTensor(train_data[:, :-1])
        y_train = torch.LongTensor(train_data[:, -1])
        X_test = torch.FloatTensor(test_data[:, :-1])
        y_test = torch.LongTensor(test_data[:, -1])
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        bottom_model = ShadowBottomModel(input_dim=X_train.shape[1]).to(device)
        top_model = ShadowTopModel().to(device)
        
        optimizer_bottom = optim.Adam(bottom_model.parameters(), lr=0.001)
        optimizer_top = optim.Adam(top_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(num_epochs):
            bottom_model.train()
            top_model.train()
            
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer_bottom.zero_grad()
                optimizer_top.zero_grad()
                
                embeddings = bottom_model(inputs)
                outputs = top_model(embeddings)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer_bottom.step()
                optimizer_top.step()
        
        # Evaluation on test set
        bottom_model.eval()
        top_model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                embeddings = bottom_model(inputs)
                outputs = top_model(embeddings)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Shadow Model {i} Test Performance:")
        print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
        
        # Collect data for attack model
        with torch.no_grad():
            for loader, label in [(train_loader, 1), (test_loader, 0)]:
                for inputs, _ in loader:
                    inputs = inputs.to(device)
                    embeddings = bottom_model(inputs)
                    outputs = top_model(embeddings)
                    
                    for emb, out in zip(embeddings.cpu().numpy(), outputs.cpu().numpy()):
                        if label == 1:
                            attack_train_data.append(np.concatenate([emb, out, [label]]))
                        else:
                            attack_test_data.append(np.concatenate([emb, out, [label]]))
    
    np.save(f'attack_model_data/attack_train_data.npy', np.array(attack_train_data))
    np.save(f'attack_model_data/attack_test_data.npy', np.array(attack_test_data))

if __name__ == "__main__":
    train_shadow_models()
