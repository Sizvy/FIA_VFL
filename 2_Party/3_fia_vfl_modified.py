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
from final_vfl import BottomModel

class ResBlock(nn.Module):
    """Residual block for reconstructor"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(dim)
        )
    
    def forward(self, x):
        return x + self.block(x)

class EnhancedFeatureInferenceAttacker(nn.Module):
    """Enhanced feature inference attack with improved architecture and training"""
    def __init__(self, active_dim=24, passive_dim=24, emb_dim=32):
        super().__init__()
        
        # Initialize logger
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = f'results/attack_results_{timestamp}.txt'
        
        with open(self.log_file, 'w') as f:
            f.write("VFL Feature Inference Attack Results\n")
            f.write("="*50 + "\n")
            f.write(f"Start Time: {datetime.now()}\n\n")
            f.write("Model Architecture:\n")
            f.write(f"- Active Dim: {active_dim}\n")
            f.write(f"- Passive Dim: {passive_dim}\n")
            f.write(f"- Embedding Dim: {emb_dim}\n")
            f.write("="*50 + "\n\n")
        
        # Feature discriminator with spectral normalization
        self.discriminator = nn.Sequential(
            spectral_norm(nn.Linear(emb_dim * 2, 256)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            spectral_norm(nn.Linear(128, 1)),
            nn.Sigmoid()
        )
        
        # Feature reconstructor with residual connections
        self.reconstructor = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            ResBlock(256),
            ResBlock(256),
            nn.Linear(256, passive_dim)
        )
        
        self.criterion = nn.BCELoss()
        self.recon_criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # For visualization
        self.train_losses = []
        self.val_metrics = []
    
    def forward(self, emb_active, emb_passive):
        combined = torch.cat([emb_active, emb_passive], dim=1)
        return self.discriminator(combined)
    
    def reconstruct(self, emb_passive):
        return self.reconstructor(emb_passive)
    
    def _log_message(self, message, print_msg=True):
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
        if print_msg:
            print(message)
    
    def train_model(self, active_data, passive_data, active_embs, passive_embs, 
                  epochs=500, batch_size=512, recon_weight=0.3):
        """Enhanced training with visualization"""
        # Log training parameters
        self._log_message("\nTraining Parameters:", print_msg=False)
        self._log_message(f"- Epochs: {epochs}", print_msg=False)
        self._log_message(f"- Batch Size: {batch_size}", print_msg=False)
        self._log_message(f"- Recon Weight: {recon_weight}", print_msg=False)
        self._log_message("-"*50)
        
        # Create dataset with proper shuffling
        idx = np.random.permutation(len(active_embs))
        X = {
            'active_data': np.vstack([active_data, active_data]),
            'passive_data': np.vstack([passive_data, passive_data[idx]]),
            'active_emb': np.vstack([active_embs, active_embs]),
            'passive_emb': np.vstack([passive_embs, passive_embs[idx]]),
            'labels': np.concatenate([np.ones(len(active_embs)), np.zeros(len(active_embs))])
        }
        
        # Split data
        (X_train, X_val) = train_test_split(
            np.arange(len(X['labels'])), test_size=0.2, random_state=42)
        
        # Create loaders
        train_loader = self._create_loader(X, X_train, batch_size, shuffle=True)
        val_loader = self._create_loader(X, X_val, batch_size*2, shuffle=False)
        
        # Visualize embeddings before training
        self._visualize_embeddings(active_embs, passive_embs, "Before Training")
        
        best_val_auc = 0.5
        for epoch in range(epochs):
            self.train()  # Set model to training mode
            epoch_loss = 0
            preds, labels = [], []
            
            for batch in train_loader:
                x_active, x_passive, emb_active, emb_passive, y = batch
                self.optimizer.zero_grad()
                
                # Forward pass
                disc_out = self(emb_active, emb_passive)
                recon_out = self.reconstruct(emb_passive)
                
                # Losses
                disc_loss = self.criterion(disc_out.squeeze(), y)
                recon_loss = self.recon_criterion(recon_out, x_passive)
                loss = disc_loss + recon_weight * recon_loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                preds.extend(disc_out.detach().squeeze().tolist())
                labels.extend(y.tolist())
            
            # Validation
            val_metrics = self._validate(val_loader, recon_weight)
            self.scheduler.step()
            
            # Save best model
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                torch.save(self.state_dict(), 'best_attack_model.pt')
            
            # Store metrics
            train_auc = roc_auc_score(labels, preds)
            self.train_losses.append(epoch_loss/len(train_loader))
            self.val_metrics.append(val_metrics)
            
            # Log progress
            if (epoch+1) % 20 == 0:
                log_msg = f"Epoch {epoch+1}/{epochs} | Loss: {self.train_losses[-1]:.4f} | " \
                          f"Train AUC: {train_auc:.4f} | Val AUC: {val_metrics['auc']:.4f} | " \
                          f"Val Recon: {val_metrics['recon_error']:.4f}"
                self._log_message(log_msg)
        
        # Visualize results
        self._plot_training_curve()
        self._visualize_embeddings(active_embs, passive_embs, "After Training")
        
        return best_val_auc
    
    def _create_loader(self, X, indices, batch_size, shuffle):
        dataset = TensorDataset(
            torch.FloatTensor(X['active_data'][indices]),
            torch.FloatTensor(X['passive_data'][indices]),
            torch.FloatTensor(X['active_emb'][indices]),
            torch.FloatTensor(X['passive_emb'][indices]),
            torch.FloatTensor(X['labels'][indices])
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _validate(self, loader, recon_weight):
        self.eval()
        preds, labels = [], []
        recon_errors = []
        
        with torch.no_grad():
            for batch in loader:
                x_active, x_passive, emb_active, emb_passive, y = batch
                # Discriminator task
                outputs = self(emb_active, emb_passive)
                preds.extend(outputs.squeeze().tolist())
                labels.extend(y.tolist())
                
                # Reconstruction task
                recon = self.reconstruct(emb_passive)
                recon_errors.append(self.recon_criterion(recon, x_passive).item())
        
        metrics = self._calculate_metrics(labels, preds)
        metrics['recon_error'] = np.mean(recon_errors)
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred):
        y_pred_class = (np.array(y_pred) > 0.5).astype(int)
        return {
            'auc': roc_auc_score(y_true, y_pred),
            'ap': average_precision_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred_class),
            'precision': precision_score(y_true, y_pred_class),
            'recall': recall_score(y_true, y_pred_class),
            'f1': f1_score(y_true, y_pred_class),
            'cm': confusion_matrix(y_true, y_pred_class),
            'report': classification_report(y_true, y_pred_class, 
                                         target_names=['Surrogate', 'Real']),
            'recon_error': None  # Will be set in _validate
        }
    
    def _visualize_embeddings(self, active_embs, passive_embs, title):
        """Visualize embeddings using PCA"""
        combined = np.vstack([active_embs, passive_embs])
        pca = PCA(n_components=2).fit_transform(combined)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(pca[:len(active_embs), 0], pca[:len(active_embs), 1], 
                   alpha=0.3, label='Active', c='blue')
        plt.scatter(pca[len(active_embs):, 0], pca[len(active_embs):, 1], 
                   alpha=0.3, label='Passive', c='red')
        plt.title(f"Embedding Visualization - {title}")
        plt.legend()
        plt.savefig(f"embeddings_{title.lower().replace(' ', '_')}.png")
        plt.close()
    
    def _plot_training_curve(self):
        """Plot training and validation metrics"""
        plt.figure(figsize=(12, 4))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.title("Training Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        # AUC
        plt.subplot(1, 2, 2)
        val_aucs = [m['auc'] for m in self.val_metrics]
        plt.plot(val_aucs, label='Val AUC', color='orange')
        plt.title("Validation AUC")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        
        plt.tight_layout()
        plt.savefig("training_curve.png")
        plt.close()

def run_enhanced_attack():
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
    
    # Analyze embeddings
    attacker = EnhancedFeatureInferenceAttacker(
        active_dim=client1_train.shape[1],
        passive_dim=client2_train.shape[1],
        emb_dim=active_embs.shape[1]
    ).to(device)
    
    attacker._log_message("\nEmbedding Analysis:")
    attacker._log_message(f"Active embeddings - Mean: {active_embs.mean():.4f}, Std: {active_embs.std():.4f}")
    attacker._log_message(f"Passive embeddings - Mean: {passive_embs.mean():.4f}, Std: {passive_embs.std():.4f}")
    
    # Train attack model
    attacker._log_message("\nTraining enhanced attack model...")
    best_auc = attacker.train_model(
        active_data=client1_train,
        passive_data=client2_train,
        active_embs=active_embs,
        passive_embs=passive_embs,
        epochs=500,
        batch_size=512,
        recon_weight=0.3
    )
    
    # Final evaluation with surrogates
    attacker._log_message("\nRunning final evaluation with surrogates...")
    attacker._log_message("="*60)
    attacker._log_message("Final Evaluation with Surrogates")
    attacker._log_message("="*60)
    
    # Create test set with surrogates
    surrogate_idx = np.random.permutation(len(passive_embs))
    test_data = {
        'active_data': np.vstack([client1_train, client1_train]),
        'passive_data': np.vstack([client2_train, client2_train[surrogate_idx]]),
        'active_emb': np.vstack([active_embs, active_embs]),
        'passive_emb': np.vstack([passive_embs, passive_embs[surrogate_idx]]),
        'labels': np.concatenate([np.ones(len(active_embs)), np.zeros(len(active_embs))])
    }
    
    test_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(test_data['active_data']),
            torch.FloatTensor(test_data['passive_data']),
            torch.FloatTensor(test_data['active_emb']),
            torch.FloatTensor(test_data['passive_emb']),
            torch.FloatTensor(test_data['labels'])
        ),
        batch_size=512
    )
    
    # Load best model and evaluate
    attacker.load_state_dict(torch.load('best_attack_model.pt'))
    metrics = attacker._validate(test_loader, recon_weight=0.3)
    
    # Log final results
    attacker._log_message(f"AUC-ROC: {metrics['auc']:.4f}")
    attacker._log_message(f"Reconstruction MSE: {metrics['recon_error']:.4f}\n")
    attacker._log_message("Classification Report:")
    attacker._log_message(metrics['report'])
    attacker._log_message("="*60)
    attacker._log_message(f"\nBest Validation AUC: {best_auc:.4f}")
    attacker._log_message(f"End Time: {datetime.now()}")
    
    print(f"\nAll results saved to: {attacker.log_file}")

if __name__ == "__main__":
    run_enhanced_attack()
