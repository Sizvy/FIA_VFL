import torch
import numpy as np
from models.discriminator_targeted_feature import FeatureDiscriminator
from torch.utils.data import DataLoader, TensorDataset

class FeatureInferenceAttack:
    def __init__(self, target_feature_idx, emb_dim, device='cuda'):
        self.target_feature_idx = target_feature_idx
        self.discriminator = FeatureDiscriminator(emb_dim*2).to(device)
        self.device = device
        self.optimizer = None
        self.criterion = None

    def train(self, active_embs, passive_embs, active_raw, epochs=200, batch_size=256):
        unique_values = np.unique(active_raw[:, self.target_feature_idx])
        labels = np.array([1 if x in unique_values else 0 for x in active_raw[:, self.target_feature_idx]])
        
        # Class-weighted loss
        pos_weight = torch.tensor([len(labels)/sum(labels)-1]).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        dataset = TensorDataset(
            torch.cat([torch.FloatTensor(active_embs), 
                     torch.FloatTensor(passive_embs)], dim=1),
            torch.FloatTensor(labels)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
        self.discriminator.train()
        epoch_loss = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.discriminator(inputs).squeeze()
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss/len(loader)

    def infer(self, active_emb, passive_emb):
        with torch.no_grad():
            combined = torch.cat([
                torch.FloatTensor(active_emb), 
                torch.FloatTensor(passive_emb)
            ], dim=1).to(self.device)
            prob = self.discriminator(combined).item()
            return prob > 0.5, prob
