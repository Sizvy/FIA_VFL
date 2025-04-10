import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                            recall_score, f1_score, confusion_matrix, 
                            average_precision_score, classification_report)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.nn.utils import spectral_norm
from datetime import datetime
import os
from final_vfl_5 import BottomModel

class Discriminator(nn.Module):
    """Standalone discriminator to distinguish real vs shuffled passive embeddings"""
    def __init__(self, emb_dim):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Linear(emb_dim * 2, 256)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            spectral_norm(nn.Linear(128, 1)),
            nn.Sigmoid()
        )
    
    def forward(self, emb_active, emb_passive):
        combined = torch.cat([emb_active, emb_passive], dim=1)
        return self.model(combined)

class EmbeddingDiscriminatorAttack:
    """Attack using only discriminator to detect embedding correlations"""
    def __init__(self, active_dim=24, passive_dim=24, emb_dim=32):
        super().__init__()
        
        # Initialize logger
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = f'results/discriminator_results_{timestamp}.txt'
        
        with open(self.log_file, 'w') as f:
            f.write("VFL Embedding Discriminator Attack Results\n")
            f.write("="*50 + "\n")
            f.write(f"Start Time: {datetime.now()}\n\n")
            f.write("Model Architecture:\n")
            f.write(f"- Active Dim: {active_dim}\n")
            f.write(f"- Passive Dim: {passive_dim}\n")
            f.write(f"- Embedding Dim: {emb_dim}\n")
            f.write("="*50 + "\n\n")
        
        # Initialize discriminator only
        self.discriminator = Discriminator(emb_dim)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(self.discriminator.parameters(), 
                                          lr=1e-4, weight_decay=1e-5)
        
        # For visualization
        self.train_losses = []
        self.val_metrics = []
    
    def _log_message(self, message, print_msg=True):
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
        if print_msg:
            print(message)

    def train_model(self, active_embs, passive_embs, epochs=500, batch_size=512):
        """Train discriminator to detect correlated embeddings"""
        # Create dataset: label=1 for real pairs, label=0 for shuffled pairs
        idx = np.random.permutation(len(active_embs))
        X = {
            'active_emb': np.vstack([active_embs, active_embs]),
            'passive_emb': np.vstack([passive_embs, passive_embs[idx]]),
            'labels': np.concatenate([np.ones(len(active_embs)), 
                                    np.zeros(len(active_embs))])
        }
        
        # Split data
        (X_train, X_val) = train_test_split(
            np.arange(len(X['labels'])), test_size=0.2, random_state=42)
        
        # Create loaders
        train_loader = self._create_loader(X, X_train, batch_size, shuffle=True)
        val_loader = self._create_loader(X, X_val, batch_size*2, shuffle=False)
        
        best_val_auc = 0.5
        for epoch in range(epochs):
            self.discriminator.train()
            epoch_loss = 0
            preds, labels = [], []
            
            for batch in train_loader:
                emb_active, emb_passive, y = batch
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.discriminator(emb_active, emb_passive).squeeze()
                loss = self.criterion(outputs, y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                preds.extend(outputs.detach().tolist())
                labels.extend(y.tolist())
            
            # Validation
            val_metrics = self._validate(val_loader)
            self.train_losses.append(epoch_loss/len(train_loader))
            
            # Save best model
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                torch.save(self.discriminator.state_dict(), 'best_discriminator.pt')
            
            # Log progress
            if (epoch+1) % 20 == 0:
                log_msg = f"Epoch {epoch+1}/{epochs} | Loss: {self.train_losses[-1]:.4f} | " \
                          f"Val AUC: {val_metrics['auc']:.4f}"
                self._log_message(log_msg)
        
        self._plot_training_curve()
        return best_val_auc
    
    def _create_loader(self, X, indices, batch_size, shuffle):
        dataset = TensorDataset(
            torch.FloatTensor(X['active_emb'][indices]),
            torch.FloatTensor(X['passive_emb'][indices]),
            torch.FloatTensor(X['labels'][indices])
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _validate(self, loader):
        self.discriminator.eval()
        preds, labels = [], []
        
        with torch.no_grad():
            for emb_active, emb_passive, y in loader:
                outputs = self.discriminator(emb_active, emb_passive).squeeze()
                preds.extend(outputs.tolist())
                labels.extend(y.tolist())
        
        return {
            'auc': roc_auc_score(labels, preds),
            'ap': average_precision_score(labels, preds),
            'accuracy': accuracy_score(labels, (np.array(preds) > 0.5).astype(int))
        }
    
    def _plot_training_curve(self):
        """Plot training loss curve"""
        plt.figure(figsize=(6, 4))
        plt.plot(self.train_losses, label='Train Loss')
        plt.title("Training Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("training_curve.png")
        plt.close()

def run_discriminator_attack():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    client1_train = np.load('splitted_data/client_1_train.npy')
    client2_train = np.load('splitted_data/client_2_train.npy')
    
    # Load models
    checkpoint = torch.load('Models/best_vfl_model.pt')
    
    # Initialize models
    active_bottom = BottomModel(input_dim=client1_train.shape[1]).to(device)
    passive_bottom = BottomModel(input_dim=client2_train.shape[1]).to(device)
    active_bottom.load_state_dict(checkpoint['client1_bottom'])
    passive_bottom.load_state_dict(checkpoint['client2_bottom'])
    active_bottom.eval()
    passive_bottom.eval()
    
    # Get embeddings
    with torch.no_grad():
        active_embs = active_bottom(torch.FloatTensor(client1_train).to(device)).cpu().numpy()
        passive_embs = passive_bottom(torch.FloatTensor(client2_train).to(device)).cpu().numpy()
    
    # Initialize and run attack
    attacker = EmbeddingDiscriminatorAttack(
        active_dim=client1_train.shape[1],
        passive_dim=client2_train.shape[1],
        emb_dim=active_embs.shape[1]
    )
    best_auc = attacker.train_model(active_embs, passive_embs, epochs=500)
    
    print(f"\nBest Validation AUC: {best_auc:.4f}")
    print(f"Results saved to: {attacker.log_file}")

if __name__ == "__main__":
    run_discriminator_attack()
