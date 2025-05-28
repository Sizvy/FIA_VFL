import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt  # For visualization
from averageBottom import BottomModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
DISTRIBUTION_TYPE = "kde"  # "gaussian" or "kde"

SHADOW_PLUS_F_MODEL = "../shadow_model_data/shadow_plus_F_client2_bottom.pt"
SHADOW_MINUS_F_MODEL = "../shadow_model_data/shadow_minus_F_client2_bottom.pt"

def load_shadow_model(model_path, input_dim):
    model = BottomModel(input_dim=input_dim, output_dim=64).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def extract_embeddings(model, X_data):
    dataset = TensorDataset(torch.FloatTensor(X_data))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch[0].to(device)
            emb = model(batch).cpu().numpy()
            embeddings.append(emb)
    return np.concatenate(embeddings, axis=0)

def fit_distribution(embeddings):
    if DISTRIBUTION_TYPE == "gaussian":
        mean = np.mean(embeddings, axis=0)
        cov = np.cov(embeddings.T)
        return {"type": "gaussian", "mean": mean, "cov": cov}
    elif DISTRIBUTION_TYPE == "kde":
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(embeddings)
        return {"type": "kde", "kde": kde}

def plot_embeddings(E_plus_F, E_minus_F):
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    all_embeddings = np.concatenate([E_plus_F, E_minus_F], axis=0)
    pca.fit(all_embeddings)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(*pca.transform(E_plus_F).T, label="With F", alpha=0.5)
    plt.scatter(*pca.transform(E_minus_F).T, label="Without F", alpha=0.5)
    plt.title("Embedding Distribution (PCA Reduced)")
    plt.legend()
    plt.savefig("../shadow_model_data/embedding_distributions.png")
    plt.close()

if __name__ == "__main__":
    X_plus_F = np.load("../shadow_model_data/shadow_plus_F_client_2_train.npy")
    X_minus_F = np.load("../shadow_model_data/shadow_minus_F_client_2_train.npy")
    
    shadow_plus_F = load_shadow_model(SHADOW_PLUS_F_MODEL, X_plus_F.shape[1])
    shadow_minus_F = load_shadow_model(SHADOW_MINUS_F_MODEL, X_minus_F.shape[1])
    
    print("Extracting embeddings for D+F...")
    E_plus_F = extract_embeddings(shadow_plus_F, X_plus_F)
    
    print("Extracting embeddings for D-F...")
    E_minus_F = extract_embeddings(shadow_minus_F, X_minus_F)
    
    print(f"Embedding shapes: With F {E_plus_F.shape}, Without F {E_minus_F.shape}")
    
    print("\nFitting distributions...")
    P_E_plus_F = fit_distribution(E_plus_F)
    P_E_minus_F = fit_distribution(E_minus_F)
    
    np.savez("../shadow_model_data/embedding_distributions.npz",
             P_E_plus_F=P_E_plus_F,
             P_E_minus_F=P_E_minus_F)
    
    plot_embeddings(E_plus_F, E_minus_F)
    
    print("""
    Step 3 Complete! Results saved:
    - Embedding distributions: ../shadow_model_data/embedding_distributions.npz
    - Visualization: ../shadow_model_data/embedding_distributions.png
    """)
