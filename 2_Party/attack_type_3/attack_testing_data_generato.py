import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from models.averageBottom import BottomModel
from models.simpleTop import TopModel
from data.data_loader import load_client_data, create_dataloaders
from training.train_utils import train_one_epoch
from training.validation_utils import validate


os.makedirs('attack_model_data', exist_ok=True)
def get_all_outputs(models, loaders1, loaders2, device):
    """Get outputs for all data splits"""
    client1_bottom, client2_bottom, top_model = models
    client1_bottom.eval()
    client2_bottom.eval()
    top_model.eval()
    
    all_outputs = []
    all_labels = []
    all_membership = []
    
    # Process each split (train=0, val=1, test=2)
    for split_idx, (loader1, loader2) in enumerate(zip(loaders1, loaders2)):
        for batch1, batch2 in zip(loader1, loader2):
            # Handle different batch formats
            data1 = batch1[0] if isinstance(batch1, (tuple, list)) else batch1
            data2 = batch2[0] if isinstance(batch2, (tuple, list)) else batch2
            
            # Get labels if available (only for client1)
            labels = batch1[1] if isinstance(batch1, (tuple, list)) and len(batch1) > 1 else None
            
            data1, data2 = data1.to(device), data2.to(device)
            
            with torch.no_grad():
                out1 = client1_bottom(data1)
                out2 = client2_bottom(data2)
                h_combined = torch.cat([out1, out2], dim=1)
                outputs = top_model(h_combined)
                # probs = torch.softmax(outputs, dim=1)
                # prob_vec = probs.cpu().numpy()
            # Get labels (only available from client1)
            if labels is not None:
                labels = labels.cpu().numpy()
            else:
                # This shouldn't happen since client1 has labels
                raise ValueError("Labels not found in client1 data")
            
            # Membership: 1 for train (split_idx=0), 0 for others
            membership = np.ones(len(labels)) if split_idx == 0 else np.zeros(len(labels))
            
            all_outputs.append(outputs)
            all_labels.append(labels)
            all_membership.append(membership)
    
    return np.vstack(all_outputs), np.concatenate(all_labels), np.concatenate(all_membership)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50 
    patience = 5    
    batch_size = 128
    
    # Load data using your existing loader
    client1_data = load_client_data(1)  # Returns (train, val, test, y_train, y_val, y_test)
    client2_data = load_client_data(2)  # Returns (train, val, test)
    
    print(f"Client 1 samples - Train: {len(client1_data[0])}, Val: {len(client1_data[1])}, Test: {len(client1_data[2])}")
    print(f"Client 2 samples - Train: {len(client2_data[0])}, Val: {len(client2_data[1])}, Test: {len(client2_data[2])}")
    
    # Create dataloaders (using your existing create_dataloaders)
    train_loader1, val_loader1, test_loader1 = create_dataloaders(
        *client1_data, batch_size=batch_size
    )
    train_loader2, val_loader2, test_loader2 = create_dataloaders(
        *client2_data, batch_size=batch_size
    )

    # Initialize models
    client1_bottom = BottomModel(input_dim=client1_data[0].shape[1], output_dim=64).to(device)
    client2_bottom = BottomModel(input_dim=client2_data[0].shape[1], output_dim=64).to(device)
    top_model = TopModel().to(device)

    # Training setup (unchanged)
    optimizer1 = optim.AdamW(client1_bottom.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer2_bottom = optim.Adam(client2_bottom.parameters(), lr=0.001)
    optimizer_top = optim.Adam(top_model.parameters(), lr=0.001)

    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=num_epochs)
    scheduler2_bottom = optim.lr_scheduler.CosineAnnealingLR(optimizer2_bottom, T_max=num_epochs)
    scheduler_top = optim.lr_scheduler.CosineAnnealingLR(optimizer_top, T_max=num_epochs)
 
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    counter = 0
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            train_loader1, train_loader2,
            client1_bottom, client2_bottom, top_model,
            criterion, optimizer1, optimizer2_bottom, optimizer_top, device
        )
        
        val_loss, val_acc, val_f1 = validate(
            val_loader1, val_loader2,
            [client1_bottom, client2_bottom, top_model],
            criterion, device
        )
        
        scheduler1.step()
        scheduler2_bottom.step()
        scheduler_top.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save({
                'client1_bottom': client1_bottom.state_dict(),
                'client2_bottom': client2_bottom.state_dict(),
                'top_model': top_model.state_dict()
            }, 'Saved_Models/best_vfl_model.pt')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    # Load best model
    checkpoint = torch.load('Saved_Models/best_vfl_model.pt')
    client1_bottom.load_state_dict(checkpoint['client1_bottom'])
    client2_bottom.load_state_dict(checkpoint['client2_bottom'])
    top_model.load_state_dict(checkpoint['top_model'])
    
    # Get outputs for ALL data (train + val + test)
    outputs, labels, membership = get_all_outputs(
        [client1_bottom, client2_bottom, top_model],
        [train_loader1, val_loader1, test_loader1],
        [train_loader2, val_loader2, test_loader2],
        device
    )
    
    # Combine into final array: [outputs, labels, membership]
    attack_data = np.column_stack([outputs, labels, membership])
    
    np.save('attack_model_data/testing_outputs.npy', attack_data)
    
    print("\nVictim model outputs saved successfully:")
    print(f"- Total samples: {len(attack_data)}")
    print(f"- Training samples (membership=1): {np.sum(membership == 1)}")
    print(f"- Val/Test samples (membership=0): {np.sum(membership == 0)}")
    print(f"- Output vector shape: {outputs.shape[1]}-dimensional")
    print(f"- Saved to: attack_data/victim_outputs.npy")

if __name__ == "__main__":
    main()
