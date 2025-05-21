import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from attack_model_2 import EnhancedAttackModel

# Load your EnhancedAttackModel class definition here (same as in your training script)

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('attack_model_data/test_confusion_matrix.png')
    plt.close()

def test_attack_model():
    # Load victim model outputs
    victim_outputs = np.load('attack_model_data/victim_outputs.npy')
    
    # Prepare features and labels
    X = np.column_stack([
        victim_outputs[:, :-2],  # Prediction vectors
        victim_outputs[:, -2]    # Original class labels
    ])
    y = victim_outputs[:, -1]    # Membership labels
    
    print(f"\nLoaded victim outputs:")
    print(f"- Total samples: {len(X)}")
    print(f"- 'In' samples (training data): {np.sum(y == 1)}")
    print(f"- 'Out' samples (val/test data): {np.sum(y == 0)}")
    
    # Create test dataset
    test_dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Initialize attack model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedAttackModel(input_dim=X.shape[1]).to(device)
    model.load_state_dict(torch.load('Saved_Models/best_attack_model.pt'))
    model.eval()
    
    # Testing
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            
            all_probs.extend(outputs.cpu().numpy())
            all_preds.extend((outputs > 0.5).float().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\nAttack Model Test Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Save results
    results = {
        'accuracy': acc,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'true_labels': all_labels
    }
    np.save('attack_model_data/test_results.npy', results)

if __name__ == "__main__":
    test_attack_model()
