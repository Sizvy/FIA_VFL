import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import matplotlib.pyplot as plt
from models.discriminator import Discriminator
from utils.metrics import calculate_metrics

class EmbeddingDiscriminatorAttack:
    def __init__(self, active_dim, passive_dim, emb_dim):
        self.discriminator = Discriminator(emb_dim)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=1e-4, weight_decay=1e-5)
        self.train_losses = []

    def train_model(self, active_embs, passive_embs, epochs=200, batch_size=512):
        X = self._prepare_data(active_embs, passive_embs)
        train_loader, val_loader = self._create_data_loaders(X, batch_size)
        
        best_metrics = None
        for epoch in range(epochs):
            self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)
            
            if best_metrics is None or val_metrics['auc'] > best_metrics['auc']:
                best_metrics = val_metrics
            
            if (epoch+1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Val AUC: {val_metrics['auc']:.4f}")
        
        print("\n=== Best Validation Metrics ===")
        self._print_metrics(best_metrics)
        return best_metrics

    def _prepare_data(self, active_embs, passive_embs):
        idx = np.random.permutation(len(active_embs))
        return {
            'active_emb': np.vstack([active_embs, active_embs]),
            'passive_emb': np.vstack([passive_embs, passive_embs[idx]]),
            'labels': np.concatenate([np.ones(len(active_embs)), np.zeros(len(active_embs))])
        }

    def _create_data_loaders(self, X, batch_size):
        train_idx, val_idx = train_test_split(np.arange(len(X['labels'])), test_size=0.2)
        
        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X['active_emb'][train_idx]),
                torch.FloatTensor(X['passive_emb'][train_idx]),
                torch.FloatTensor(X['labels'][train_idx])
            ),
            batch_size=batch_size, 
            shuffle=True
        )
        
        val_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X['active_emb'][val_idx]),
                torch.FloatTensor(X['passive_emb'][val_idx]),
                torch.FloatTensor(X['labels'][val_idx])
            ),
            batch_size=batch_size*2, 
            shuffle=False
        )
        
        return train_loader, val_loader

    def _train_epoch(self, loader):
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

    def _validate(self, loader):
        self.discriminator.eval()
        preds, labels = [], []
        with torch.no_grad():
            for emb_active, emb_passive, y in loader:
                outputs = self.discriminator(emb_active, emb_passive).squeeze()
                preds.extend(outputs.tolist())
                labels.extend(y.tolist())
        return calculate_metrics(np.array(preds), np.array(labels))

    def _print_metrics(self, metrics):
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
