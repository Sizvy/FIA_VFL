import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt 
import seaborn as sns
from averageBottom import BottomModel
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
DISTRIBUTION_TYPE = "kde"

SHADOW_PLUS_F_MODEL = "../shadow_model_data/shadow_plus_F_client2_bottom.pt"
SHADOW_MINUS_F_MODEL = "../shadow_model_data/shadow_minus_F_client2_bottom.pt"

SHADOW_PLUS_F_MODEL_client1 = "../shadow_model_data/shadow_plus_F_client1_bottom.pt"
SHADOW_MINUS_F_MODEL_client1 = "../shadow_model_data/shadow_minus_F_client1_bottom.pt"

SHADOW_PLUS_F_MODEL_inter = "../shadow_model_data/shadow_plus_F_client2_bottom_inter.pt"
SHADOW_MINUS_F_MODEL_inter = "../shadow_model_data/shadow_minus_F_client2_bottom_inter.pt"

SHADOW_PLUS_F_MODEL_client1_inter = "../shadow_model_data/shadow_plus_F_client1_bottom_inter.pt"
SHADOW_MINUS_F_MODEL_client1_inter = "../shadow_model_data/shadow_minus_F_client1_bottom_inter.pt"

def load_shadow_model(model_path, input_dim):
    model = BottomModel(input_dim=input_dim, output_dim=64).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def extract_embeddings_1(model, X_data):
    dataset = TensorDataset(torch.FloatTensor(X_data))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch[0].to(device)
            emb = model(batch).cpu().numpy()
            embeddings.append(emb)
    return np.concatenate(embeddings, axis=0)

def extract_embeddings_2(model, X_data):
    dataset = TensorDataset(torch.FloatTensor(X_data))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch[0].to(device)
            logits = model(batch)
            confidences = torch.sigmoid(logits).cpu().numpy()
            logit_scaled = np.log(confidences / (1 - confidences + 1e-10))
            all_embeddings.append(logit_scaled)

    full_embeddings = np.concatenate(all_embeddings, axis=0)
    return full_embeddings

def extract_embeddings_3(model, X_data):
    dataset = TensorDataset(torch.FloatTensor(X_data))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    hybrids = []
    with torch.no_grad():
        for batch in loader:
            batch = batch[0].to(device)
            raw = model(batch)
            conf = torch.sigmoid(raw)
            logit = torch.log(conf / (1 - conf + 1e-10))
            # Combine raw and logit features
            hybrid = torch.cat([raw, logit], dim=1)
            hybrids.append(hybrid.cpu().numpy())
    return np.concatenate(hybrids, axis=0)



def calculate_distances(E_plus_F, E_minus_F, E_plus_F_inter, E_minus_F_inter):
    # Ensure both embeddings have the same shape
    assert E_plus_F.shape == E_minus_F.shape, "Embeddings must have the same shape!"
    
    # Euclidean Distance (L2)
    l2_distances = np.sqrt(np.sum((E_plus_F - E_minus_F) ** 2, axis=1))
    mean_l2 = np.mean(l2_distances)
    
    # Manhattan Distance (L1)
    l1_distances = np.sum(np.abs(E_plus_F - E_minus_F), axis=1)
    mean_l1 = np.mean(l1_distances)
    
    # Cosine Distance
    dot_product = np.sum(E_plus_F * E_minus_F, axis=1)
    norm_plus = np.linalg.norm(E_plus_F, axis=1)
    norm_minus = np.linalg.norm(E_minus_F, axis=1)
    cosine_similarity = dot_product / (norm_plus * norm_minus + 1e-10)  # Avoid division by zero
    cosine_distance = 1 - cosine_similarity
    mean_cosine = np.mean(cosine_distance)


    ### For Inter
    # Ensure both embeddings have the same shape
    assert E_plus_F_inter.shape == E_minus_F_inter.shape, "Embeddings must have the same shape!"

    # Euclidean Distance (L2)
    l2_distances_inter = np.sqrt(np.sum((E_plus_F_inter - E_minus_F_inter) ** 2, axis=1))
    mean_l2_inter = np.mean(l2_distances_inter)

    # Manhattan Distance (L1)
    l1_distances_inter = np.sum(np.abs(E_plus_F_inter - E_minus_F_inter), axis=1)
    mean_l1_inter = np.mean(l1_distances_inter)

    # Cosine Distance
    dot_product_inter = np.sum(E_plus_F_inter * E_minus_F_inter, axis=1)
    norm_plus_inter = np.linalg.norm(E_plus_F_inter, axis=1)
    norm_minus_inter = np.linalg.norm(E_minus_F_inter, axis=1)
    cosine_similarity_inter = dot_product_inter / (norm_plus_inter * norm_minus_inter + 1e-10)  # Avoid division by zero
    cosine_distance_inter = 1 - cosine_similarity_inter
    mean_cosine_inter = np.mean(cosine_distance_inter)
    
    return {
        "mean_euclidean_distance": mean_l2,
        "mean_manhattan_distance": mean_l1,
        "mean_cosine_distance": mean_cosine,
        "mean_euclidean_distance_inter": mean_l2_inter,
        "mean_manhattan_distance_inter": mean_l1_inter,
        "mean_cosine_distance_inter": mean_cosine_inter
    }


def calculate_distances_for_plotting(E_plus_F, E_minus_F, E_plus_F_inter, E_minus_F_inter):
    # Intra-sample distances
    intra_euclidean = np.linalg.norm(E_plus_F - E_minus_F, axis=1)
    intra_manhattan = np.sum(np.abs(E_plus_F - E_minus_F), axis=1)
    intra_cosine = 1 - np.sum(E_plus_F * E_minus_F, axis=1) / (np.linalg.norm(E_plus_F, axis=1) * np.linalg.norm(E_minus_F, axis=1))
    
    # Inter-sample distances
    inter_euclidean = np.linalg.norm(E_plus_F_inter - E_minus_F_inter, axis=1)
    inter_manhattan = np.sum(np.abs(E_plus_F_inter - E_minus_F_inter), axis=1)
    inter_cosine = 1 - np.sum(E_plus_F_inter * E_minus_F_inter, axis=1) / (np.linalg.norm(E_plus_F_inter, axis=1) * np.linalg.norm(E_minus_F_inter, axis=1))
    
    return {
        'intra': {'euclidean': intra_euclidean, 'manhattan': intra_manhattan, 'cosine': intra_cosine},
        'inter': {'euclidean': inter_euclidean, 'manhattan': inter_manhattan, 'cosine': inter_cosine}
    }

def plot_ablation_results(results_df):
    plt.figure(figsize=(10,6))
    sns.barplot(x='num_features', y='roc_auc', hue='features', 
                data=results_df, dodge=False)
    plt.title("Ablation Study: Feature Combinations vs ROC AUC")
    plt.xlabel("Number of Features")
    plt.ylabel("ROC AUC Score")
    plt.legend(title='Feature Combination')
    plt.tight_layout()
    plt.savefig('../shadow_model_data/ablation_studyi.png')
    plt.show()

def run_ablation_study(distances_dict):
    from classify_distances import train_distance_classifier
    feature_combinations = [
        ['euclidean'],
        ['manhattan'],
        ['cosine'],
        ['euclidean', 'manhattan'],
        ['euclidean', 'cosine'],
        ['manhattan', 'cosine'],
        ['euclidean', 'manhattan', 'cosine']  # All features
    ]
    
    results = []
    for features in feature_combinations:
        print(f"\n=== Testing features: {features} ===")
        clf, report, roc_auc = train_distance_classifier(
            distances_dict,
            features=features,
            model_type='svm'  # Use logistic for fair comparison
        )
        
        # Store results
        result = {
            'features': features,
            'roc_auc': roc_auc,
            'report': report,
            'num_features': len(features)
        }
        results.append(result)
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    X_plus_F = np.load("../shadow_model_data/shadow_plus_F_client_2_train.npy")
    X_minus_F = np.load("../shadow_model_data/shadow_minus_F_client_2_train.npy")

    X_plus_F_client1 = np.load("../shadow_model_data/shadow_plus_F_client_1_train.npy")
    X_minus_F_client1 = np.load("../shadow_model_data/shadow_minus_F_client_1_train.npy")
    
    shadow_plus_F = load_shadow_model(SHADOW_PLUS_F_MODEL, X_plus_F.shape[1])
    shadow_minus_F = load_shadow_model(SHADOW_MINUS_F_MODEL, X_minus_F.shape[1])

    shadow_plus_F_client1 = load_shadow_model(SHADOW_PLUS_F_MODEL_client1, X_plus_F_client1.shape[1])
    shadow_minus_F_client1 = load_shadow_model(SHADOW_MINUS_F_MODEL_client1, X_minus_F_client1.shape[1])
    
    print("Extracting embeddings for D+F...")
    E_plus_F = extract_embeddings_2(shadow_plus_F, X_plus_F)
    E_plus_F_client1 = extract_embeddings_2(shadow_plus_F_client1, X_plus_F_client1)
    E_plus_F = np.concatenate([E_plus_F, E_plus_F_client1], axis=1)
    
    print("Extracting embeddings for D-F...")
    E_minus_F = extract_embeddings_2(shadow_minus_F, X_minus_F)
    E_minus_F_client1 = extract_embeddings_2(shadow_plus_F_client1, X_plus_F_client1)
    E_minus_F = np.concatenate([E_minus_F, E_minus_F_client1], axis=1)
    
    print(f"Embedding shapes: With F {E_plus_F.shape}, Without F {E_minus_F.shape}")


    ### For inter

    X_plus_F_inter = np.load("../shadow_model_data/shadow_plus_F_client_2_train_inter.npy")
    X_minus_F_inter = np.load("../shadow_model_data/shadow_minus_F_client_2_train_inter.npy")

    X_plus_F_client1_inter = np.load("../shadow_model_data/shadow_plus_F_client_1_train_inter.npy")
    X_minus_F_client1_inter = np.load("../shadow_model_data/shadow_minus_F_client_1_train_inter.npy")

    shadow_plus_F_inter = load_shadow_model(SHADOW_PLUS_F_MODEL_inter, X_plus_F_inter.shape[1])
    shadow_minus_F_inter = load_shadow_model(SHADOW_MINUS_F_MODEL_inter, X_minus_F_inter.shape[1])

    shadow_plus_F_client1_inter = load_shadow_model(SHADOW_PLUS_F_MODEL_client1_inter, X_plus_F_client1_inter.shape[1])
    shadow_minus_F_client1_inter = load_shadow_model(SHADOW_MINUS_F_MODEL_client1_inter, X_minus_F_client1_inter.shape[1])

    print("Extracting embeddings for D+F...")
    E_plus_F_inter = extract_embeddings_2(shadow_plus_F_inter, X_plus_F_inter)
    E_plus_F_client1_inter = extract_embeddings_2(shadow_plus_F_client1_inter, X_plus_F_client1_inter)
    E_plus_F_inter = np.concatenate([E_plus_F_inter, E_plus_F_client1_inter], axis=1)

    print("Extracting embeddings for D-F...")
    E_minus_F_inter = extract_embeddings_2(shadow_minus_F_inter, X_minus_F_inter)
    E_minus_F_client1_inter = extract_embeddings_2(shadow_plus_F_client1_inter, X_plus_F_client1_inter)
    E_minus_F_inter = np.concatenate([E_minus_F_inter, E_minus_F_client1_inter], axis=1)

    print(f"Embedding shapes: With F {E_plus_F_inter.shape}, Without F {E_minus_F_inter.shape}")



    # Calculate distances
    distances = calculate_distances(E_plus_F, E_minus_F, E_plus_F_inter, E_minus_F_inter)
    print("Average Distances Between Embeddings(Intra):")
    print(f"- Euclidean (L2): {distances['mean_euclidean_distance']:.4f}")
    print(f"- Manhattan (L1): {distances['mean_manhattan_distance']:.4f}")
    print(f"- Cosine: {distances['mean_cosine_distance']:.4f}")
    print("Average Distances Between Embeddings(Inter):")
    print(f"- Euclidean (L2): {distances['mean_euclidean_distance_inter']:.4f}")
    print(f"- Manhattan (L1): {distances['mean_manhattan_distance_inter']:.4f}")
    print(f"- Cosine: {distances['mean_cosine_distance_inter']:.4f}")
    
    distances_dict = calculate_distances_for_plotting(E_plus_F, E_minus_F, E_plus_F_inter, E_minus_F_inter)
    
    # Run ablation study
    ablation_results = run_ablation_study(distances_dict)
    
    # Save and display results
    ablation_results.to_csv('../shadow_model_data/ablation_results.csv', index=False)
    print(ablation_results[['features', 'roc_auc']])
    
    print("""
    Step 3 Complete! Results saved:
    - Embedding distributions: ../shadow_model_data/embedding_distributions.npz
    - Classifier decision boundaries: ../shadow_model_data/*_decision_boundary.png
    """)


