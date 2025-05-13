import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.aux_attack_model import EnhancedDiscriminator

class FeatureInferenceAttack:
    def __init__(self, target_feature_idx, emb_dim, device='cuda', known_samples=None):
        self.target_feature_idx = target_feature_idx
        self.emb_dim = emb_dim
        self.device = device
        self.known_samples = known_samples  # Dict: {feature_value: (active_emb, passive_emb)}
        
        self.discriminator = EnhancedDiscriminator(emb_dim * 2).to(device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.reg_criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=0.001, weight_decay=1e-4)

    def train(self, combined_embs, labels, feature_values=None, epochs=200, batch_size=256):
        # Multi-task loss weights
        lambda_disc = 0.7
        lambda_reg = 0.3
        
        # Prepare known samples if available
        known_active, known_passive, known_values = None, None, None
        if self.known_samples:
            known_active = torch.stack([v[0] for v in self.known_samples.values()]).to(self.device)
            known_passive = torch.stack([v[1] for v in self.known_samples.values()]).to(self.device)
            known_values = torch.FloatTensor(list(self.known_samples.keys())).to(self.device)
        
        # Create dataset
        dataset = TensorDataset(
            torch.FloatTensor(combined_embs),
            torch.FloatTensor(labels),
            torch.FloatTensor(feature_values) if feature_values is not None else torch.zeros(len(combined_embs))
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0
            for inputs, labels, values in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                is_real_pred, value_pred = self.discriminator(inputs)
                
                # Discriminator loss
                disc_loss = self.criterion(is_real_pred.squeeze(), labels)
                
                # Feature value regression loss (only on known samples)
                reg_loss = 0
                if self.known_samples:
                    known_combined = torch.cat([known_active, known_passive], dim=1)
                    _, known_value_pred = self.discriminator(known_combined)
                    reg_loss = self.reg_criterion(known_value_pred, known_values)
                
                # Combined loss
                total_loss = lambda_disc * disc_loss + lambda_reg * reg_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
            
            scheduler.step()
            
            # Early stopping
            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    def infer(self, active_emb, passive_emb):
        """Returns: (is_real_prediction, predicted_feature_value)"""
        self.discriminator.eval()
        with torch.no_grad():
            active_tensor = torch.FloatTensor(active_emb).to(self.device)
            passive_tensor = torch.FloatTensor(passive_emb).to(self.device)
            
            if len(active_tensor.shape) == 1:
                active_tensor = active_tensor.unsqueeze(0)
            if len(passive_tensor.shape) == 1:
                passive_tensor = passive_tensor.unsqueeze(0)
            
            combined = torch.cat([active_tensor, passive_tensor], dim=1)
            is_real, pred_value = self.discriminator(combined)
            
            return is_real.item() > 0.5, pred_value.item()
