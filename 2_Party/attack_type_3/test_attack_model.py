import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from attack_model_2 import EnhancedAttackModel
from tqdm import tqdm

def run_single_trial(trial_num=None):
    # Load victim model outputs
    victim_outputs = np.load('attack_model_data/testing_outputs.npy')
    
    # Prepare features and labels
    X = np.column_stack([
        victim_outputs[:, :-2],  # Prediction vectors
        victim_outputs[:, -2]    # Original class labels
    ])
    y = victim_outputs[:, -1]    # Membership labels
    
    # Create test dataset with shuffling
    test_dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)  # Shuffle for different splits
    
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
    
    # if trial_num is not None:
        # plot_confusion_matrix(cm, classes=['Out', 'In'], trial=trial_num)
    
    return {
        'accuracy': acc,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'true_labels': all_labels
    }

def test_attack_model_multiple_trials(num_trials=20):
    # Initialize metrics storage
    all_metrics = {
        'accuracy': [],
        'f1_score': [],
        'auc': [],
        'confusion_matrices': []
    }
    
    # Run multiple trials
    for trial in tqdm(range(num_trials), desc="Running trials"):
        results = run_single_trial(trial_num=trial+1)
        
        # Store metrics
        all_metrics['accuracy'].append(results['accuracy'])
        all_metrics['f1_score'].append(results['f1_score'])
        all_metrics['auc'].append(results['auc'])
        all_metrics['confusion_matrices'].append(results['confusion_matrix'])
    
    # Calculate average metrics
    avg_metrics = {
        'mean_accuracy': np.mean(all_metrics['accuracy']),
        'std_accuracy': np.std(all_metrics['accuracy']),
        'mean_f1': np.mean(all_metrics['f1_score']),
        'std_f1': np.std(all_metrics['f1_score']),
        'mean_auc': np.mean(all_metrics['auc']),
        'std_auc': np.std(all_metrics['auc']),
        'mean_confusion_matrix': np.mean(all_metrics['confusion_matrices'], axis=0),
        'all_trials': all_metrics
    }
    
    # Print results
    print("\nAverage Attack Model Test Results (20 trials):")
    print(f"Accuracy: {avg_metrics['mean_accuracy']:.4f} ± {avg_metrics['std_accuracy']:.4f}")
    print(f"F1 Score: {avg_metrics['mean_f1']:.4f} ± {avg_metrics['std_f1']:.4f}")
    print(f"AUC: {avg_metrics['mean_auc']:.4f} ± {avg_metrics['std_auc']:.4f}")
    print("Average Confusion Matrix:")
    print(avg_metrics['mean_confusion_matrix'].astype(int))
    
    # Plot average confusion matrix
    # plot_confusion_matrix(avg_metrics['mean_confusion_matrix'].astype(int), classes=['Out', 'In'])
    
    # Save results
    # np.save('attack_model_data/average_test_results.npy', avg_metrics)
    return avg_metrics

if __name__ == "__main__":
    # Run single trial (original behavior)
    # test_attack_model()  
    
    # Run multiple trials
    final_results = test_attack_model_multiple_trials(num_trials=20)
