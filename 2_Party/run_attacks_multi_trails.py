import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                            recall_score, f1_score, confusion_matrix, 
                            average_precision_score, classification_report)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.nn.utils import spectral_norm
from datetime import datetime
import os
from models.averageBottom import BottomModel
from attacks.embedding_discriminator import EmbeddingDiscriminatorAttack

def run_single_trial(trial_num, client1_train, client2_train, emb_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    active_bottom = BottomModel(input_dim=client1_train.shape[1]).to(device)
    passive_bottom = BottomModel(input_dim=client2_train.shape[1]).to(device)

    # Load pretrained weights
    checkpoint = torch.load('Saved_Models/best_vfl_model.pt')
    active_bottom.load_state_dict(checkpoint['client1_bottom'], strict=False)
    passive_bottom.load_state_dict(checkpoint['client2_bottom'], strict=False)

    active_bottom.eval()
    passive_bottom.eval()

    # Get all embeddings
    with torch.no_grad():
        active_embs = active_bottom(torch.FloatTensor(client1_train).to(device)).cpu().numpy()
        passive_embs = passive_bottom(torch.FloatTensor(client2_train).to(device)).cpu().numpy()

    # Initialize and run attack
    attacker = EmbeddingDiscriminatorAttack(
        active_dim=client1_train.shape[1],
        passive_dim=client2_train.shape[1],
        emb_dim=emb_dim
    )
    
    # Set random seeds for reproducibility
    torch.manual_seed(42 + trial_num)
    np.random.seed(42 + trial_num)
    
    best_auc = attacker.train_model(active_embs, passive_embs, epochs=500)
    
    # Get final metrics from the attacker
    final_metrics = {
        'auc': best_auc,
        'accuracy': attacker.val_metrics[-1]['accuracy'],
        'f1': attacker.val_metrics[-1]['f1'],
        'precision': attacker.val_metrics[-1]['precision'],
        'recall': attacker.val_metrics[-1]['recall']
    }
    
    print(f"\nTrial {trial_num+1} Best Validation AUC: {best_auc:.4f}")
    print(f"Results saved to: {attacker.log_file}")
    
    return final_metrics

def run_discriminator_attack():
    # Load data once (shared across all trials)
    client1_train = np.load('splitted_data/client_1_train.npy')
    client2_train = np.load('splitted_data/client_2_train.npy')

    # Determine embedding dimension once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        test_input = torch.FloatTensor(client1_train[:1]).to(device)
        emb_dim = BottomModel(input_dim=client1_train.shape[1])(test_input).shape[1]

    # Run 10 trials
    trial_metrics = []
    for trial_num in range(10):
        print(f"\n=== Starting Trial {trial_num+1}/10 ===")
        metrics = run_single_trial(trial_num, client1_train, client2_train, emb_dim)
        trial_metrics.append(metrics)

    # Calculate statistics across trials
    auc_values = [m['auc'] for m in trial_metrics]
    accuracies = [m['accuracy'] for m in trial_metrics]
    f1_scores = [m['f1'] for m in trial_metrics]
    precisions = [m['precision'] for m in trial_metrics]
    recalls = [m['recall'] for m in trial_metrics]

    # Print final summary in requested format
    print("\n=== Final Results ===")
    print(f"AUC: {np.mean(auc_values):.4f} ± {np.std(auc_values):.4f}")
    print(f"ACCURACY: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"PRECISION: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"RECALL: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")

if __name__ == "__main__":
    run_discriminator_attack()
