import os
import torch
import torch.optim as optim
import torch.nn as nn
from opacus import PrivacyEngine

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")
from models.complexBottom import BottomModel
from models.simpleTop import TopModel
from data.data_loader import load_client_data, create_dataloaders
from training.train_utils import train_one_epoch
from training.validation_utils import validate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 25 
    patience = 5
    batch_size = 128
    
    # Load and verify data
    client1_data = load_client_data(1)
    client2_data = load_client_data(2)
    
    print(f"Client 1 samples: {len(client1_data[0])}")
    print(f"Client 2 samples: {len(client2_data[0])}")
    
    # Create dataloaders
    train_loader1, val_loader1, test_loader1 = create_dataloaders(
        *client1_data, batch_size=batch_size
    )
    train_loader2, val_loader2, test_loader2 = create_dataloaders(
        *client2_data[:3], batch_size=batch_size
    )

    # Initialize models
    client1_bottom = BottomModel(input_dim=client1_data[0].shape[1], output_dim=64).to(device)
    client2_bottom = BottomModel(input_dim=client2_data[0].shape[1], output_dim=64).to(device)
    top_model = TopModel().to(device)

    # Enhanced optimizers with weight decay
    optimizer1 = optim.AdamW(client1_bottom.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer2_bottom = optim.Adam(client2_bottom.parameters(), lr=0.001)
    optimizer_top = optim.Adam(top_model.parameters(), lr=0.001)

    # Privacy engine with adjusted parameters
    privacy_engine = PrivacyEngine()
    client1_bottom, optimizer1, train_loader1 = privacy_engine.make_private(
        module=client1_bottom,
        optimizer=optimizer1,
        data_loader=train_loader1,
        noise_multiplier=1.2,
        max_grad_norm=1.0,
        poisson_sampling=False
    )

    # For Client 2
    privacy_engine2 = PrivacyEngine()
    client2_bottom, optimizer2, train_loader2 = privacy_engine2.make_private(
        module=client2_bottom,
        optimizer=optimizer2_bottom,
        data_loader=train_loader2,
        noise_multiplier=1.2,
        max_grad_norm=1.0,
        poisson_sampling=False
    )

    # Print noise statistics
    # sigma = 1.2 * 1.0
    # variance = sigma ** 2
    # print("\nNoise Parameters:")
    # print(f"Standard deviation (σ): {sigma:.4f}")
    # print(f"Variance (σ²): {variance:.4f}")
    # print(f"Applied per parameter in gradient updates\n")

    # Learning rate scheduler
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=num_epochs)
    scheduler2_bottom = optim.lr_scheduler.CosineAnnealingLR(optimizer2_bottom, T_max=num_epochs)
    scheduler_top = optim.lr_scheduler.CosineAnnealingLR(optimizer_top, T_max=num_epochs)

    # Label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    counter = 0

    # Training loop with early stopping based on accuracy
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
        
        # print(f"Epoch {epoch+1}/{num_epochs}:")
        # print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        # print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        # print(f"Current LR: {optimizer1.param_groups[0]['lr']:.6f}")

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

    # Final evaluation
    checkpoint = torch.load('Saved_Models/best_vfl_model.pt')
    client1_bottom.load_state_dict(checkpoint['client1_bottom'])
    client2_bottom.load_state_dict(checkpoint['client2_bottom'])
    top_model.load_state_dict(checkpoint['top_model'])
    
    test_loss, test_acc, test_f1 = validate(
        test_loader1, test_loader2,
        [client1_bottom, client2_bottom, top_model],
        criterion, device
    )

    print("\nFinal Results:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Privacy Budget (ε): {privacy_engine.get_epsilon(delta=1e-5):.2f}")

if __name__ == "__main__":
    main()
