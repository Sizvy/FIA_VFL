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
from utils.logger import log_message
from utils.metrics import calculate_metrics

class EmbeddingDiscriminatorAttack:
    def __init__(self, active_dim, passive_dim, emb_dim):
        self._init_logger()
        self.discriminator = Discriminator(emb_dim)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(
            self.discriminator.parameters(), 
            lr=1e-4, 
            weight_decay=1e-5
        )
        self.train_losses = []
        self.val_metrics = []

    def _init_logger(self):
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = f'results/discriminator_results_{timestamp}.txt'
        log_message("VFL Embedding Discriminator Attack Results", self.log_file)
        log_message(f"Start Time: {datetime.now()}\n", self.log_file)

    def train_model(self, active_embs, passive_embs, epochs=500, batch_size=512):
        X = self._prepare_data(active_embs, passive_embs)
        train_loader, val_loader = self._create_data_loaders(X, batch_size)
        
        best_val_auc = 0.5
        for epoch in range(epochs):
            self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)
            
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                torch.save(self.discriminator.state_dict(), 'best_discriminator.pt')
            
            if (epoch+1) % 20 == 0:
                self._log_epoch(epoch, epochs, val_metrics)
        
        self._finalize_training(val_loader)
        return best_val_auc

    def _prepare_data(self, active_embs, passive_embs):
        idx = np.random.permutation(len(active_embs))
        return {
            'active_emb': np.vstack([active_embs, active_embs]),
            'passive_emb': np.vstack([passive_embs, passive_embs[idx]]),
            'labels': np.concatenate([np.ones(len(active_embs)), 
                                    np.zeros(len(active_embs))])
        }

    def _create_data_loaders(self, X, batch_size):
        train_idx, val_idx = train_test_split(
            np.arange(len(X['labels'])), test_size=0.2, random_state=42)
        
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

    def _log_epoch(self, epoch, epochs, metrics):
        log_msg = f"Epoch {epoch+1}/{epochs} | Loss: {self.train_losses[-1]:.4f}\n"
        log_msg += f"Val AUC: {metrics['auc']:.4f} | Accuracy: {metrics['accuracy']:.4f}"
        log_message(log_msg, self.log_file)

    def _finalize_training(self, val_loader):
        self.discriminator.load_state_dict(torch.load('best_discriminator.pt'))
        final_metrics = self._validate(val_loader)
        log_message(f"\nFinal AUC: {final_metrics['auc']:.4f}", self.log_file)
        self._plot_training_curve()

    def _plot_training_curve(self):
        plt.figure(figsize=(6, 4))
        plt.plot(self.train_losses)
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("training_curve.png")
        plt.close()
