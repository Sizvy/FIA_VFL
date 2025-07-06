import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
from averageBottom import BottomModel
from simpleTop import TopModel
from sklearn.decomposition import PCA

# ===== CONFIG =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
DISTRIBUTION_TYPE = "kde"

TARGET_MODEL_PATH = "../Saved_Models/best_vfl_model.pt"
VICTIM_DATA_PATH = "../splitted_data/client_2_train.npy"
VICTIM_DATA_PATH_client1 = "../splitted_data/client_1_train.npy"
DISTRIBUTIONS_PATH = "../shadow_model_data/embedding_distributions.npz"

# ===== LOAD TARGET MODEL =====
def load_target_model(client1_dim, client2_dim):
    checkpoint = torch.load(TARGET_MODEL_PATH)
    
    client1_bottom = BottomModel(input_dim=client1_dim, output_dim=64).to(device)
    client2_bottom = BottomModel(input_dim=client2_dim, output_dim=64).to(device)
    top_model = TopModel().to(device)
    
    client1_bottom.load_state_dict(checkpoint['client1_bottom'])
    client2_bottom.load_state_dict(checkpoint['client2_bottom'])
    top_model.load_state_dict(checkpoint['top_model'])
    
    client2_bottom.eval()
    client1_bottom.eval()
    return client2_bottom, client1_bottom

# ===== LOAD DISTRIBUTIONS FROM STEP 3 =====
def load_distributions():
    data = np.load(DISTRIBUTIONS_PATH, allow_pickle=True)
    return data["P_E_plus_F"].item(), data["P_E_minus_F"].item()

# ===== QUERY TARGET MODEL =====
def query_target_embeddings_1(model, X_data):
    dataset = TensorDataset(torch.FloatTensor(X_data))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch[0].to(device)
            emb = model(batch).cpu().numpy()
            embeddings.append(emb)
    return np.concatenate(embeddings, axis=0)

def query_target_embeddings_2(model, X_data):
    dataset = TensorDataset(torch.FloatTensor(X_data))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Step 1: Collect all embeddings
    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch[0].to(device)
            logits = model(batch)
            confidences = torch.sigmoid(logits).cpu().numpy()
            logit_scaled = np.log(confidences / (1 - confidences + 1e-10))
            all_embeddings.append(logit_scaled)
    
    # Step 2: Concatenate and apply PCA
    full_embeddings = np.concatenate(all_embeddings, axis=0)
    
    return full_embeddings

# ===== MEMBERSHIP INFERENCE =====
def compute_log_likelihood_1(emb, distribution):
    if distribution["type"] == "gaussian":
        return multivariate_normal.logpdf(emb, mean=distribution["mean"], cov=distribution["cov"])
    elif distribution["type"] == "kde":
        return distribution["kde"].score_samples(emb.reshape(1, -1))[0]

def run_attack_1(target_embeddings, P_E_plus_F, P_E_minus_F):
    results = []
    for emb in target_embeddings:
        score_plus = compute_log_likelihood_1(emb, P_E_plus_F)
        score_minus = compute_log_likelihood_1(emb, P_E_minus_F)
        # print(f"\n{score_plus}-{score_minus}\n")
        results.append(score_plus > score_minus)
    return np.mean(results)

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    X_victim = np.load(VICTIM_DATA_PATH)
    X_victim_client1 = np.load(VICTIM_DATA_PATH_client1)
    print(f"Victim data shape: {X_victim.shape}")
    
    client2_bottom, client1_bottom  = load_target_model(
        client1_dim=X_victim_client1.shape[1],
        client2_dim=X_victim.shape[1]
    )
    
    target_embeddings = query_target_embeddings_2(client2_bottom, X_victim)
    helper_embeddings = query_target_embeddings_2(client1_bottom, X_victim_client1)
    target_embeddings = np.concatenate([target_embeddings, helper_embeddings], axis=1)
    print(f"Extracted {len(target_embeddings)} target embeddings")

    P_E_plus_F, P_E_minus_F = load_distributions()
    
    attack_success_rate = run_attack_1(target_embeddings, P_E_plus_F, P_E_minus_F)
    
    print(f"\nAttack Results:")
    print(f"Probability that feature F exists in victim data: {attack_success_rate:.2%}")
    print("Interpretation:")
    print("- Close to 50%: Attack failed (can't distinguish)")
    print("- >70%: Strong evidence F exists in victim data")
    print("- <30%: Strong evidence F is missing from victim data")
