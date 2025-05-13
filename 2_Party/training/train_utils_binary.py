import torch
import torch.nn as nn

def train_one_epoch(client1_loader, client2_loader, 
                   client1_bottom, client2_bottom, top_model, 
                   criterion, optimizer1, optimizer2_bottom, optimizer_top, device):
    """Training function for binary classification with BCEWithLogitsLoss"""
    client1_bottom.train()
    client2_bottom.train()
    top_model.train()
    
    running_loss = 0.0
    for (x1, y), (x2,) in zip(client1_loader, client2_loader):
        # Convert y to float and ensure proper device placement
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True)  # Changed for BCE
        
        batch_size = min(x1.size(0), x2.size(0), y.size(0))
        x1 = x1[:batch_size]
        x2 = x2[:batch_size]
        y = y[:batch_size]

        # Forward pass
        optimizer1.zero_grad()
        optimizer2_bottom.zero_grad()
        optimizer_top.zero_grad()
        
        h1 = client1_bottom(x1)
        h2 = client2_bottom(x2)
        h_combined = torch.cat([h1, h2], dim=1)
        outputs = top_model(h_combined).squeeze()  # Squeeze for single output
        
        # Backward pass with gradient clipping
        loss = criterion(outputs, y)
        if torch.isnan(loss).any():
            print("NaN detected in loss! Skipping batch")
            optimizer1.zero_grad()
            continue
            
        loss.backward()
        
        # Gradient clipping for all models
        torch.nn.utils.clip_grad_norm_(client1_bottom.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(client2_bottom.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(top_model.parameters(), 1.0)
        
        optimizer1.step()
        optimizer2_bottom.step()
        optimizer_top.step()
        
        running_loss += loss.item()
    
    return running_loss / len(client1_loader)
