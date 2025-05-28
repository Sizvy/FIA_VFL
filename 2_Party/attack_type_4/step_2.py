import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from averageBottom import BottomModel
from simpleTop import TopModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.001

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

# ===== SHADOW MODEL TRAINING =====
def train_shadow_model(X1, X2, y, client2_has_F):
    client1_bottom = BottomModel(input_dim=X1.shape[1], output_dim=64).to(device)
    client2_bottom = BottomModel(input_dim=X2.shape[1], output_dim=64).to(device)
    top_model = TopModel().to(device)
    
    optimizer = optim.Adam(
        list(client1_bottom.parameters()) + 
        list(client2_bottom.parameters()) + 
        list(top_model.parameters()), 
        lr=LR
    )
    criterion = nn.CrossEntropyLoss()
    loader = create_dataloaders(X1, X2, y, BATCH_SIZE)
    
    for epoch in range(EPOCHS):
        client1_bottom.train()
        client2_bottom.train()
        top_model.train()
        
        for batch_X1, batch_X2, batch_y in loader:
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")
    
    return client2_bottom

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    X1_plus_F, X2_plus_F, y_plus_F = load_shadow_data(SHADOW_PLUS_F_PATH)
    X1_minus_F, X2_minus_F, y_minus_F = load_shadow_data(SHADOW_MINUS_F_PATH)
    
    print("Training shadow model with F...")
    shadow_client2_plus_F = train_shadow_model(X1_plus_F, X2_plus_F, y_plus_F, client2_has_F=True)
    
    print("\nTraining shadow model without F...")
    shadow_client2_minus_F = train_shadow_model(X1_minus_F, X2_minus_F, y_minus_F, client2_has_F=False)
    
    torch.save(shadow_client2_plus_F.state_dict(), f"{SHADOW_PLUS_F_PATH}_client2_bottom.pt")
    torch.save(shadow_client2_minus_F.state_dict(), f"{SHADOW_MINUS_F_PATH}_client2_bottom.pt")
    print("\nShadow models saved for embedding analysis.")
