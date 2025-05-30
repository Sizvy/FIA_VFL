import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
from averageBottom import BottomModel
from simpleTop import TopModel

# ===== CONFIG =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
DISTRIBUTION_TYPE = "kde"

TARGET_MODEL_PATH = "../Saved_Models/best_vfl_model.pt"
VICTIM_DATA_PATH = "../splitted_data/client_2_test.npy"  
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
    return client2_bottom

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

    logit_scaled = []
    with torch.no_grad():
        for batch in loader:
            batch = batch[0].to(device)
            logits = model(batch)  # Raw logits
            confidences = torch.sigmoid(logits).cpu().numpy()  # Sigmoid
            logit_scaled.append(np.log(confidences / (1 - confidences + 1e-10)))  # Logit scaling
    return np.concatenate(logit_scaled, axis=0)

def query_target_embeddings_3(model, X_data):
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

# ===== MEMBERSHIP INFERENCE =====
def compute_log_likelihood_1(emb, distribution):
    if distribution["type"] == "gaussian":
        return multivariate_normal.logpdf(emb, mean=distribution["mean"], cov=distribution["cov"])
    elif distribution["type"] == "kde":
        return distribution["kde"].score_samples(emb.reshape(1, -1))[0]

def run_attack(target_embeddings, P_E_plus_F, P_E_minus_F):
    results = []
    for emb in target_embeddings:
        score_plus = compute_log_likelihood_1(emb, P_E_plus_F)
        score_minus = compute_log_likelihood_1(emb, P_E_minus_F)
        # print(f"\n{score_plus}-{score_minus}\n")
        results.append(score_plus > score_minus)
    return np.mean(results)

def compute_log_likelihood_2(emb, distribution):
    if "hybrid" in distribution["type"]:
        # Split the input embedding
        split_point = distribution.get("feature_split", emb.shape[0]//2)
        raw = emb[:split_point]
        logit = emb[split_point:]

        if "gaussian" in distribution["type"]:
            logp_raw = multivariate_normal.logpdf(
                raw,
                mean=distribution["raw_mean"],
                cov=distribution["raw_cov"]
            )
            logp_logit = multivariate_normal.logpdf(
                logit,
                mean=distribution["logit_mean"],
                cov=distribution["logit_cov"]
            )
        else:  # KDE
            logp_raw = distribution["kde_raw"].score_samples(raw.reshape(1, -1))[0]
            logp_logit = distribution["kde_logit"].score_samples(logit.reshape(1, -1))[0]

        return logp_raw + logp_logit  # Combined log-likelihood

    # Original non-hybrid case
    elif distribution["type"] == "gaussian":
        return multivariate_normal.logpdf(emb, mean=distribution["mean"], cov=distribution["cov"])
    elif distribution["type"] == "kde":
        return distribution["kde"].score_samples(emb.reshape(1, -1))[0]

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    X_victim = np.load(VICTIM_DATA_PATH)
    print(f"Victim data shape: {X_victim.shape}")
    
    client2_bottom = load_target_model(
        client1_dim=np.load("../splitted_data/client_1_test.npy").shape[1],
        client2_dim=X_victim.shape[1]
    )
    
    target_embeddings = query_target_embeddings_2(client2_bottom, X_victim)
    print(f"Extracted {len(target_embeddings)} target embeddings")

    P_E_plus_F, P_E_minus_F = load_distributions()
    
    attack_success_rate = run_attack(target_embeddings, P_E_plus_F, P_E_minus_F)
    
    print(f"\nAttack Results:")
    print(f"Probability that feature F exists in victim data: {attack_success_rate:.2%}")
    print("Interpretation:")
    print("- Close to 50%: Attack failed (can't distinguish)")
    print("- >70%: Strong evidence F exists in victim data")
    print("- <30%: Strong evidence F is missing from victim data")
