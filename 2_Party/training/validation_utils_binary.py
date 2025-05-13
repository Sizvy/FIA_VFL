import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def validate(client1_loader, client2_loader, models, criterion, device):
    client1_bottom, client2_bottom, top_model = models
    client1_bottom.eval()
    client2_bottom.eval()
    top_model.eval()
    
    val_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for (x1, y), (x2,) in zip(client1_loader, client2_loader):
            x1, x2, y = x1.to(device), x2.to(device), y.float().to(device)  # y.float() for BCE
            
            h1 = client1_bottom(x1)
            h2 = client2_bottom(x2)
            outputs = top_model(torch.cat([h1, h2], dim=1))
            
            loss = criterion(outputs.squeeze(), y)  # squeeze for single output
            val_loss += loss.item()
            
            # Get probabilities and predictions
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).long()  # Threshold at 0.5
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')  # Changed to binary
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5  # In case only one class present
    
    print(f"\nValidation Metrics:")
    print(f"AUC: {auc:.4f}")
    
    return val_loss / len(client1_loader), accuracy, f1
