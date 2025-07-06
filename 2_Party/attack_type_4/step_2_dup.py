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
PATIENCE = 5  # Early stopping patience

SHADOW_PLUS_F_PATH = "../shadow_model_data/shadow_plus_F"
SHADOW_MINUS_F_PATH = "../shadow_model_data/shadow_minus_F"

def load_shadow_data(prefix):
    X_client1 = np.load(f"{prefix}_client_1_train.npy")
    y = np.load(f"{prefix}_client_1_train_labels.npy")
    X_client2 = np.load(f"{prefix}_client_2_train.npy")
    return X_client1, X_client2, y

def create_dataloaders(X1, X2, y, batch_size):
    dataset = TensorDataset(
        torch.FloatTensor(X1), 
        torch.FloatTensor(X2), 
        torch.LongTensor(y)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_one_epoch(train_loader, client1_bottom, client2_bottom, top_model, criterion, optimizer1, optimizer2, optimizer_top, device):
    client1_bottom.train()
    client2_bottom.train()
    top_model.train()
    total_loss = 0.0

    for batch_X1, batch_X2, batch_y in train_loader:
        batch_X1, batch_X2, batch_y = (
            batch_X1.to(device),
            batch_X2.to(device),
            batch_y.to(device)
        )

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer_top.zero_grad()

        emb1 = client1_bottom(batch_X1)
        emb2 = client2_bottom(batch_X2)
        emb_combined = torch.cat([emb1, emb2], dim=1)
        logits = top_model(emb_combined)

        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer_top.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(val_loader, client1_bottom, client2_bottom, top_model, criterion, device):
    client1_bottom.eval()
    client2_bottom.eval()
    top_model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X1, batch_X2, batch_y in val_loader:
            batch_X1, batch_X2, batch_y = (
                batch_X1.to(device),
                batch_X2.to(device),
                batch_y.to(device)
            )

            emb1 = client1_bottom(batch_X1)
            emb2 = client2_bottom(batch_X2)
            emb_combined = torch.cat([emb1, emb2], dim=1)
            logits = top_model(emb_combined)

            loss = criterion(logits, batch_y)
            total_loss += loss.item()

            _, predicted = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = correct / total
    return total_loss / len(val_loader), accuracy

def train_shadow_model(X1, X2, y, client2_has_F):
    # Initialize models
    client1_bottom = BottomModel(input_dim=X1.shape[1], output_dim=64).to(device)
    client2_bottom = BottomModel(input_dim=X2.shape[1], output_dim=64).to(device)
    top_model = TopModel().to(device)

    # Optimizers with weight decay (matching victim training)
    optimizer1 = optim.AdamW(client1_bottom.parameters(), lr=LR, weight_decay=1e-4)
    optimizer2 = optim.Adam(client2_bottom.parameters(), lr=LR)
    optimizer_top = optim.Adam(top_model.parameters(), lr=LR)

    # Learning rate schedulers (CosineAnnealingLR)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=EPOCHS)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=EPOCHS)
    scheduler_top = optim.lr_scheduler.CosineAnnealingLR(optimizer_top, T_max=EPOCHS)

    criterion = nn.CrossEntropyLoss()
    train_loader = create_dataloaders(X1, X2, y, BATCH_SIZE)

    # Split data into train/val (80/20)
    train_size = int(0.8 * len(train_loader.dataset))
    val_size = len(train_loader.dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_loader.dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_val_acc = 0.0
    counter = 0

    # Training loop with early stopping
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            train_loader, client1_bottom, client2_bottom, top_model,
            criterion, optimizer1, optimizer2, optimizer_top, device
        )

        val_loss, val_acc = validate(
            val_loader, client1_bottom, client2_bottom, top_model,
            criterion, device
        )

        scheduler1.step()
        scheduler2.step()
        scheduler_top.step()

        print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            # Save the best model
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
    X1_plus_F, X2_plus_F, y_plus_F = load_shadow_data(SHADOW_PLUS_F_PATH)
    X1_minus_F, X2_minus_F, y_minus_F = load_shadow_data(SHADOW_MINUS_F_PATH)

    print("Training shadow model with F...")
    shadow_client2_plus_F, shadow_client1_plus_F = train_shadow_model(X1_plus_F, X2_plus_F, y_plus_F, client2_has_F=True)

    print("\nTraining shadow model without F...")
    shadow_client2_minus_F, shadow_client1_minus_F = train_shadow_model(X1_minus_F, X2_minus_F, y_minus_F, client2_has_F=False)

    # Save final models (not just best)
    torch.save(shadow_client2_plus_F.state_dict(), f"{SHADOW_PLUS_F_PATH}_client2_bottom.pt")
    torch.save(shadow_client2_minus_F.state_dict(), f"{SHADOW_MINUS_F_PATH}_client2_bottom.pt")
    torch.save(shadow_client1_plus_F.state_dict(), f"{SHADOW_PLUS_F_PATH}_client1_bottom.pt")
    torch.save(shadow_client1_minus_F.state_dict(), f"{SHADOW_MINUS_F_PATH}_client1_bottom.pt")

    print("\nShadow models saved for embedding analysis.")
