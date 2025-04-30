import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from models.discriminator import Discriminator

class EmbeddingDiscriminatorAttack:
    def __init__(self, active_dim, passive_dim, emb_dim):
        self.discriminator = Discriminator(emb_dim)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(self.discriminator.parameters(),
                                         lr=1e-4, weight_decay=1e-5)
        self.train_losses = []

    def prepare_data(self, active_embs, passive_embs):
        """Prepare real and shuffled pairs"""
        idx = np.random.permutation(len(active_embs))
        return {
            'active_emb': np.vstack([active_embs, active_embs]),
            'passive_emb': np.vstack([passive_embs, passive_embs[idx]]),
            'labels': np.concatenate([np.ones(len(active_embs)), np.zeros(len(active_embs))])
        }

    def create_loader(self, X, indices, batch_size, shuffle):
        """Create data loader from prepared data"""
        dataset = TensorDataset(
            torch.FloatTensor(X['active_emb'][indices]),
            torch.FloatTensor(X['passive_emb'][indices]),
            torch.FloatTensor(X['labels'][indices])
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_model(self, active_embs, passive_embs, epochs=500, batch_size=512):
        """Train the discriminator model"""
        X = self.prepare_data(active_embs, passive_embs)
        train_idx, val_idx = train_test_split(np.arange(len(X['labels'])), test_size=0.2, random_state=42)
        train_loader = self.create_loader(X, train_idx, batch_size, True)
        val_loader = self.create_loader(X, val_idx, batch_size*2, False)

        best_val_auc = 0.5
        for epoch in range(epochs):
            self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']

            if (epoch+1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Val AUC: {val_metrics['auc']:.4f}")

        return best_val_auc

    def validate_final(self, active_embs, passive_embs, batch_size=1024):
        """Run final validation on all data"""
        X = self.prepare_data(active_embs, passive_embs)
        loader = self.create_loader(X, np.arange(len(X['labels'])), batch_size, False)
        return self.validate(loader)

    def train_epoch(self, loader):
        """Train for one epoch"""
        self.discriminator.train()
        epoch_loss = 0
        for emb_active, emb_passive, y in loader:
            self.optimizer.zero_grad()
            outputs = self.discriminator(emb_active, emb_passive).squeeze()
            loss = self.criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.optimizer.step()
            epoch_loss += loss.item()
        self.train_losses.append(epoch_loss/len(loader))

    def validate(self, loader):
        """Validate model performance"""
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
            'accuracy': accuracy_score(labels_np, preds_binary),
            'precision': precision_score(labels_np, preds_binary),
            'recall': recall_score(labels_np, preds_binary),
            'f1': f1_score(labels_np, preds_binary)
        }
