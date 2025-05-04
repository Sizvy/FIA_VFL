import torch
import numpy as np
from models.discriminator_targeted_feature import FeatureDiscriminator
from torch.utils.data import DataLoader, TensorDataset

class FeatureInferenceAttack:
    def __init__(self, target_feature_idx, emb_dim, device='cuda'):
        self.target_feature_idx = target_feature_idx
        self.emb_dim = emb_dim
        self.device = device
        
        self.discriminator = FeatureDiscriminator(emb_dim * 2).to(device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = None

    def train(self, combined_embs, labels, epochs=200, batch_size=256):
        # Class-weighted loss
        pos_weight = torch.tensor([len(labels)/sum(labels) - 1]).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        # Create dataset
        dataset = TensorDataset(
            torch.FloatTensor(combined_embs),
            torch.FloatTensor(labels)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop with early stopping
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = self._train_epoch(loader)
            scheduler.step()

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    def _train_epoch(self, loader):
        """Train for one epoch"""
        self.discriminator.train()
        epoch_loss = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device).squeeze()

            self.optimizer.zero_grad()
            outputs = self.discriminator(inputs).squeeze()
            loss = self.criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
        return epoch_loss / len(loader)

    def infer(self, active_emb, passive_emb):
        """Make predictions on new data"""
        self.discriminator.eval()
        with torch.no_grad():
            active_tensor = torch.FloatTensor(active_emb).to(self.device)
            passive_tensor = torch.FloatTensor(passive_emb).to(self.device)

            if len(active_tensor.shape) == 1:
                active_tensor = active_tensor.unsqueeze(0)
            if len(passive_tensor.shape) == 1:
                passive_tensor = passive_tensor.unsqueeze(0)

            if active_tensor.shape[0] != passive_tensor.shape[0]:
                raise ValueError(f"Batch size mismatch: active {active_tensor.shape[0]} vs passive {passive_tensor.shape[0]}")

            combined = torch.cat([active_tensor, passive_tensor], dim=1)
            outputs = self.discriminator(combined).squeeze()
            prob = torch.sigmoid(outputs).item()
            return prob > 0.5, prob

