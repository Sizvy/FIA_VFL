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

def run_discriminator_attack():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    client1_train = np.load('splitted_data/client_1_train.npy')
    client2_train = np.load('splitted_data/client_2_train.npy')

    # Initialize models with YOUR architecture
    active_bottom = BottomModel(input_dim=client1_train.shape[1]).to(device)
    passive_bottom = BottomModel(input_dim=client2_train.shape[1]).to(device)

    # Get embeddings to determine actual output dimension
    with torch.no_grad():
        test_input = torch.FloatTensor(client1_train[:1]).to(device)
        emb_dim = active_bottom(test_input).shape[1]

    # Now load the pretrained weights with strict=False
    checkpoint = torch.load('Saved_Models/best_vfl_model.pt')
    active_bottom.load_state_dict(checkpoint['client1_bottom'], strict=False)
    passive_bottom.load_state_dict(checkpoint['client2_bottom'], strict=False)

    active_bottom.eval()
    passive_bottom.eval()

    # Get all embeddings
    with torch.no_grad():
        active_embs = active_bottom(torch.FloatTensor(client1_train).to(device)).cpu().numpy()
        passive_embs = passive_bottom(torch.FloatTensor(client2_train).to(device)).cpu().numpy()

    # Initialize and run attack with actual embedding dimension
    attacker = EmbeddingDiscriminatorAttack(
        active_dim=client1_train.shape[1],
        passive_dim=client2_train.shape[1],
        emb_dim=emb_dim  # Use actual embedding size from your model
    )
    best_auc = attacker.train_model(active_embs, passive_embs, epochs=500)

    print(f"\nBest Validation AUC: {best_auc:.4f}")
    print(f"Results saved to: {attacker.log_file}")

if __name__ == "__main__":
    run_discriminator_attack()
