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
from models.discriminator import Discriminator


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
                log_msg = (
                    f"Epoch {epoch+1}/{epochs} | Loss: {self.train_losses[-1]:.4f}\n"
                    f"Val Metrics:\n"
                    f"- AUC: {val_metrics['auc']:.4f} | AP: {val_metrics['ap']:.4f}\n"
                    f"- Accuracy: {val_metrics['accuracy']:.4f}\n"
                    f"- Precision: {val_metrics['precision']:.4f}\n"
                    f"- Recall: {val_metrics['recall']:.4f}\n"
                    f"- F1: {val_metrics['f1']:.4f}\n"
                    f"Confusion Matrix:\n{val_metrics['confusion_matrix']}"
                )
                self._log_message(log_msg)

        self._plot_training_curve()

        # Load best model and get final metrics
        self.discriminator.load_state_dict(torch.load('best_discriminator.pt'))
        final_metrics = self._validate(val_loader)

        # Generate classification report
        preds, labels = [], []
        with torch.no_grad():
            for emb_active, emb_passive, y in val_loader:
                outputs = self.discriminator(emb_active, emb_passive).squeeze()
                preds.extend(outputs.tolist())
                labels.extend(y.tolist())

        preds_binary = (np.array(preds) > 0.5).astype(int)
        report = classification_report(labels, preds_binary, target_names=['Shuffled', 'Real'])

        # Log final results
        self._log_message("\nFinal Validation Metrics:")
        self._log_message(f"AUC: {final_metrics['auc']:.4f}")
        self._log_message(f"Average Precision: {final_metrics['ap']:.4f}")
        self._log_message(f"Accuracy: {final_metrics['accuracy']:.4f}")
        self._log_message(f"Precision: {final_metrics['precision']:.4f}")
        self._log_message(f"Recall: {final_metrics['recall']:.4f}")
        self._log_message(f"F1 Score: {final_metrics['f1']:.4f}")
        self._log_message("\nConfusion Matrix:")
        self._log_message(str(final_metrics['confusion_matrix']))
        self._log_message("\nClassification Report:")
        self._log_message(report)

        return final_metrics['auc']

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

        preds_np = np.array(preds)
        labels_np = np.array(labels)
        preds_binary = (preds_np > 0.5).astype(int)

        return {
            'auc': roc_auc_score(labels_np, preds_np),
            'ap': average_precision_score(labels_np, preds_np),
            'accuracy': accuracy_score(labels_np, preds_binary),
            'precision': precision_score(labels_np, preds_binary),
            'recall': recall_score(labels_np, preds_binary),
            'f1': f1_score(labels_np, preds_binary),
            'confusion_matrix': confusion_matrix(labels_np, preds_binary)
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
