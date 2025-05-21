import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score

os.makedirs('attack_model_data', exist_ok=True)

class EnhancedShadowBottomModel(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=128, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class EnhancedShadowTopModel(nn.Module):
    def __init__(self, input_dim=128, num_classes=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
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
        
        # Class balancing
        class_counts = np.bincount(y_train.numpy())
        weights = 1. / class_counts[y_train]
        sampler = WeightedRandomSampler(weights, len(weights))
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        bottom_model = EnhancedShadowBottomModel(input_dim=X_train.shape[1]).to(device)
        top_model = EnhancedShadowTopModel().to(device)
        
        # Improved optimizer configuration
        optimizer = optim.AdamW([
            {'params': bottom_model.parameters()},
            {'params': top_model.parameters()}
        ], lr=0.001, weight_decay=1e-4)
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=0.01,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(num_epochs):
            bottom_model.train()
            top_model.train()
            
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                embeddings = bottom_model(inputs)
                outputs = top_model(embeddings)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bottom_model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(top_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
        
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
        
        # Add this in your shadow model training code (after evaluation)
        with torch.no_grad():
            for loader, membership_label in [(train_loader, 1), (test_loader, 0)]:
                for inputs, class_labels in loader:  # Now capturing original class labels
                    inputs = inputs.to(device)
                    embeddings = bottom_model(inputs)
                    outputs = top_model(embeddings)
            
                    for emb, pred_vec, class_label in zip(embeddings.cpu().numpy(),
                                                outputs.cpu().numpy(),
                                                class_labels.cpu().numpy()):
                        record = np.concatenate([
                            pred_vec,           # 11-dim prediction vector
                            [class_label],      # Original class label
                            [membership_label]  # 1 for 'in', 0 for 'out'
                        ])
                
                        if membership_label == 1:
                            attack_train_data.append(record)
                        else:
                            attack_test_data.append(record) 
    np.save(f'attack_model_data/attack_train_data.npy', np.array(attack_train_data))
    np.save(f'attack_model_data/attack_test_data.npy', np.array(attack_test_data))

if __name__ == "__main__":
    train_shadow_models()
