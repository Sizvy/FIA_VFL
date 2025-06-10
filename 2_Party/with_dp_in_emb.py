import os
import torch
import torch.optim as optim
import torch.nn as nn
from opacus.accountants import RDPAccountant
from models.complexBottom import BottomModel
from models.simpleTop import TopModel
from data.data_loader import load_client_data, create_dataloaders
from training.train_utils_dp import train_one_epoch
from training.validation_utils import validate

def compute_epsilon(sigma, sample_rate, epochs, delta):
    """Compute epsilon using RDP accounting"""
    accountant = RDPAccountant()
    
    for _ in range(epochs):
        accountant.step(
            noise_multiplier=sigma,
            sample_rate=sample_rate
        )
    
    epsilon, _ = accountant.get_privacy_spent(delta=delta)
    return epsilon

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50 
    patience = 5    
    batch_size = 128
    
    # DP parameters
    dp_sigma = 1.0
    dp_max_norm = 1.0       
    
    # Load and verify data
    client1_data = load_client_data(1)
    client2_data = load_client_data(2)

    delta = 1 / len(client1_data[0])
    
    print(f"Client 1 samples: {len(client1_data[0])}")
    print(f"Client 2 samples: {len(client2_data[0])}")
    
    # Create dataloaders
    train_loader1, val_loader1, test_loader1 = create_dataloaders(
        *client1_data, batch_size=batch_size
    )
    train_loader2, val_loader2, test_loader2 = create_dataloaders(
        *client2_data[:3], batch_size=batch_size
    )

    # Compute sample rate for DP accounting
    sample_rate = batch_size / len(train_loader1.dataset)
    
    # Print expected epsilon before training
    epsilon = compute_epsilon(dp_sigma, sample_rate, num_epochs, delta)
    print(f"\nExpected (ε, δ)-DP guarantee after training: ε = {epsilon:.2f}, δ = {delta}")

    # Initialize models
    client1_bottom = BottomModel(input_dim=client1_data[0].shape[1], output_dim=64).to(device)
    client2_bottom = BottomModel(input_dim=client2_data[0].shape[1], output_dim=64).to(device)
    top_model = TopModel().to(device)

    # Optimizers
    optimizer1 = optim.AdamW(client1_bottom.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer2_bottom = optim.Adam(client2_bottom.parameters(), lr=0.001)
    optimizer_top = optim.Adam(top_model.parameters(), lr=0.001)

    # Learning rate schedulers
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=num_epochs)
    scheduler2_bottom = optim.lr_scheduler.CosineAnnealingLR(optimizer2_bottom, T_max=num_epochs)
    scheduler_top = optim.lr_scheduler.CosineAnnealingLR(optimizer_top, T_max=num_epochs)
 
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    counter = 0

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            train_loader1, train_loader2,
            client1_bottom, client2_bottom, top_model,
            criterion, optimizer1, optimizer2_bottom, optimizer_top, device,
            dp_sigma=dp_sigma,
            dp_max_norm=dp_max_norm
        )
        
        val_loss, val_acc, val_f1 = validate(
            val_loader1, val_loader2,
            [client1_bottom, client2_bottom, top_model],
            criterion, device
        )
        
        scheduler1.step()
        scheduler2_bottom.step()
        scheduler_top.step()

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

    # Print final results and achieved epsilon
    final_epsilon = compute_epsilon(dp_sigma, sample_rate, epoch + 1, delta)
    print("\nFinal Results:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Achieved (ε, δ)-DP guarantee: ε = {final_epsilon:.2f}, δ = {delta}")

if __name__ == "__main__":
    main()
