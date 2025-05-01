import torch
import numpy as np
from models.discriminator_targeted_feature import FeatureDiscriminator
from torch.utils.data import DataLoader, TensorDataset

class FeatureInferenceAttack:
    def __init__(self, target_feature_idx, emb_dim, device='cuda'):
        self.target_feature_idx = target_feature_idx
        self.discriminator = FeatureDiscriminator(emb_dim*2).to(device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.discriminator.parameters())
        self.device = device

    def train(self, active_embs, passive_embs, active_raw, epochs=100, batch_size=64):
        unique_values = np.unique(active_raw[:, self.target_feature_idx])
        labels = np.array([1 if x in unique_values else 0 for x in active_raw[:, self.target_feature_idx]])
        
        dataset = TensorDataset(
            torch.cat([torch.FloatTensor(active_embs), torch.FloatTensor(passive_embs)], dim=1),
            torch.FloatTensor(labels)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for _ in range(epochs):
            self._train_epoch(loader)

    def _train_epoch(self, loader):
        self.discriminator.train()
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.discriminator(inputs).squeeze()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def infer(self, active_emb, passive_emb):
        with torch.no_grad():
            combined = torch.cat([torch.FloatTensor(active_emb), torch.FloatTensor(passive_emb)], dim=1).to(self.device)
            prob = self.discriminator(combined).item()
            return prob > 0.5, prob
