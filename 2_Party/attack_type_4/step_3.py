import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt 
from averageBottom import BottomModel
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
DISTRIBUTION_TYPE = "kde"

SHADOW_PLUS_F_MODEL = "../shadow_model_data/shadow_plus_F_client2_bottom.pt"
SHADOW_MINUS_F_MODEL = "../shadow_model_data/shadow_minus_F_client2_bottom.pt"

SHADOW_PLUS_F_MODEL_client1 = "../shadow_model_data/shadow_plus_F_client1_bottom.pt"
SHADOW_MINUS_F_MODEL_client1 = "../shadow_model_data/shadow_minus_F_client1_bottom.pt"

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

def fit_distribution_1(embeddings):
    if DISTRIBUTION_TYPE == "gaussian":
        mean = np.mean(embeddings, axis=0)
        cov = np.cov(embeddings.T)
        return {"type": "gaussian", "mean": mean, "cov": cov}
    elif DISTRIBUTION_TYPE == "kde":
        kde = KernelDensity(kernel='gaussian', bandwidth=0.7).fit(embeddings)
        # print(kde)
        return {"type": "kde", "kde": kde}


def calculate_distances(E_plus_F, E_minus_F):
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
    
    return {
        "mean_euclidean_distance": mean_l2,
        "mean_manhattan_distance": mean_l1,
        "mean_cosine_distance": mean_cosine
    }


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

    # Calculate distances
    distances = calculate_distances(E_plus_F, E_minus_F)
    print("Average Distances Between Embeddings:")
    print(f"- Euclidean (L2): {distances['mean_euclidean_distance']:.4f}")
    print(f"- Manhattan (L1): {distances['mean_manhattan_distance']:.4f}")
    print(f"- Cosine: {distances['mean_cosine_distance']:.4f}")

    print("\nFitting distributions...")
    P_E_plus_F = fit_distribution_1(E_plus_F)
    P_E_minus_F = fit_distribution_1(E_minus_F)
    
    np.savez("../shadow_model_data/embedding_distributions.npz",
             P_E_plus_F=P_E_plus_F,
             P_E_minus_F=P_E_minus_F)
    
    # plot_embeddings(E_plus_F, E_minus_F)
    
    print("""
    Step 3 Complete! Results saved:
    - Embedding distributions: ../shadow_model_data/embedding_distributions.npz
    """)


