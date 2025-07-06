import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from averageBottom import BottomModel
from simpleTop import TopModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.001
PATIENCE = 5
GRAD_CLIP = 1.0

SHADOW_PLUS_F_PATH = "../shadow_model_data/shadow_plus_F"
SHADOW_MINUS_F_PATH = "../shadow_model_data/shadow_minus_F"

def load_shadow_data(prefix):
    """Load data and create separate loaders like without_dp.py"""
    X_client1 = np.load(f"{prefix}_client_1_train.npy")
    X_client2 = np.load(f"{prefix}_client_2_train.npy") 
    y = np.load(f"{prefix}_client_1_train_labels.npy")
    
    # Create separate datasets/loaders like without_dp.py
    dataset1 = TensorDataset(torch.FloatTensor(X_client1), torch.LongTensor(y))
    dataset2 = TensorDataset(torch.FloatTensor(X_client2))
    
    loader1 = DataLoader(dataset1, batch_size=BATCH_SIZE, shuffle=True)
    loader2 = DataLoader(dataset2, batch_size=BATCH_SIZE, shuffle=True)
    
    return loader1, loader2

def train_one_epoch(loader1, loader2, models, optimizers, criterion):
    """Exact replica of without_dp.py's training"""
    client1_bottom, client2_bottom, top_model = models
    optimizer1, optimizer2, optimizer_top = optimizers
    
    client1_bottom.train()
    client2_bottom.train() 
    top_model.train()
    
    running_loss = 0.0
    for (x1, y), (x2,) in zip(loader1, loader2):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        
        # Match batch handling
        batch_size = min(x1.size(0), x2.size(0), y.size(0))
        x1 = x1[:batch_size]
        x2 = x2[:batch_size] 
        y = y[:batch_size]

        # Forward pass
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer_top.zero_grad()
        
        h1 = client1_bottom(x1)
        h2 = client2_bottom(x2)
        h_combined = torch.cat([h1, h2], dim=1)
        outputs = top_model(h_combined)
        
        # Loss and backward
        loss = criterion(outputs, y)
        if torch.isnan(loss).any():
            print("NaN loss detected, skipping batch")
            continue
            
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(client1_bottom.parameters(), GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(client2_bottom.parameters(), GRAD_CLIP) 
        torch.nn.utils.clip_grad_norm_(top_model.parameters(), GRAD_CLIP)
        
        optimizer1.step()
        optimizer2.step()
        optimizer_top.step()
        
        running_loss += loss.item()
    
    return running_loss / len(loader1)

def validate(loader1, loader2, models, criterion):
    """Validation matching without_dp.py"""
    client1_bottom, client2_bottom, top_model = models
    client1_bottom.eval()
    client2_bottom.eval()
    top_model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for (x1, y), (x2,) in zip(loader1, loader2):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            
            h1 = client1_bottom(x1)
            h2 = client2_bottom(x2)
            outputs = top_model(torch.cat([h1, h2], dim=1))
            
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    return total_loss / len(loader1), correct / total

def train_shadow_model(loader1, loader2, client2_has_F):
    """Training process identical to without_dp.py"""
    # Initialize models
    client1_bottom = BottomModel(input_dim=loader1.dataset[0][0].shape[0], output_dim=64).to(device)
    client2_bottom = BottomModel(input_dim=loader2.dataset[0][0].shape[0], output_dim=64).to(device)
    top_model = TopModel().to(device)
    
    # Optimizers (match without_dp.py exactly)
    optimizer1 = optim.AdamW(client1_bottom.parameters(), lr=LR, weight_decay=1e-4)
    optimizer2 = optim.Adam(client2_bottom.parameters(), lr=LR)
    optimizer_top = optim.Adam(top_model.parameters(), lr=LR)
    
    # LR schedulers
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=EPOCHS)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=EPOCHS) 
    scheduler_top = optim.lr_scheduler.CosineAnnealingLR(optimizer_top, T_max=EPOCHS)
    
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    counter = 0
    
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            loader1, loader2,
            [client1_bottom, client2_bottom, top_model],
            [optimizer1, optimizer2, optimizer_top],
            criterion
        )
        
        val_loss, val_acc = validate(
            loader1, loader2,
            [client1_bottom, client2_bottom, top_model],
            criterion
        )
        
        # Step schedulers
        scheduler1.step()
        scheduler2.step()
        scheduler_top.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save({
                'client1_bottom': client1_bottom.state_dict(),
                'client2_bottom': client2_bottom.state_dict(),
                'top_model': top_model.state_dict()
            }, f"{SHADOW_PLUS_F_PATH if client2_has_F else SHADOW_MINUS_F_PATH}_best.pt")
        else:
            counter += 1
            if counter >= PATIENCE:
                print("Early stopping triggered")
                break
    
    return client2_bottom, client1_bottom

if __name__ == "__main__":
    # Load data with separate loaders like without_dp.py
    plus_loader1, plus_loader2 = load_shadow_data(SHADOW_PLUS_F_PATH)
    minus_loader1, minus_loader2 = load_shadow_data(SHADOW_MINUS_F_PATH)
    
    print("Training shadow model with F...")
    shadow_client2_plus_F, shadow_client1_plus_F = train_shadow_model(plus_loader1, plus_loader2, True)
    
    print("\nTraining shadow model without F...") 
    shadow_client2_minus_F, shadow_client1_minus_F = train_shadow_model(minus_loader1, minus_loader2, False)
    
    # Save final models
    torch.save(shadow_client2_plus_F.state_dict(), f"{SHADOW_PLUS_F_PATH}_client2_bottom.pt")
    torch.save(shadow_client2_minus_F.state_dict(), f"{SHADOW_MINUS_F_PATH}_client2_bottom.pt")
    torch.save(shadow_client1_plus_F.state_dict(), f"{SHADOW_PLUS_F_PATH}_client1_bottom.pt")
    torch.save(shadow_client1_minus_F.state_dict(), f"{SHADOW_MINUS_F_PATH}_client1_bottom.pt")
