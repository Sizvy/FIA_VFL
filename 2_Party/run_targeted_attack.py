import numpy as np
import torch
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
from models.averageBottom import BottomModel
from attacks.feature_inference import FeatureInferenceAttack
from utils.evaluate import evaluate_attack

def load_data(client_num, split='train'):
    return np.load(f'splitted_data/client_{client_num}_{split}.npy')

def analyze_feature_correlations(active_embs, passive_data):
    """Analyze correlations between active embeddings and passive features"""
    correlations = []
    
    for col in range(passive_data.shape[1]):
        # Handle both continuous and categorical features
        if len(np.unique(passive_data[:, col])) > 10:  # Continuous feature
            mi = mutual_info_regression(active_embs, passive_data[:, col])
        else:  # Categorical feature
            mi = mutual_info_regression(active_embs, passive_data[:, col], discrete_features=[False]*active_embs.shape[1])
        
        correlations.append((col, np.mean(mi)))
    
    # Sort by correlation strength
    correlations.sort(key=lambda x: x[1], reverse=True)
    return correlations[:10]  # Return top 10

def create_evaluation_set(passive_train, passive_val, passive_test, target_idx):
    train_cutoff = int(0.8 * len(passive_train))
    real_values = passive_train[:train_cutoff, target_idx]
    
    # For continuous features, create meaningful negative samples
    if len(np.unique(real_values)) > 10:
        std_dev = np.std(real_values)
        synthetic_low = np.random.normal(
            loc=np.min(real_values)-3*std_dev,
            scale=std_dev/2,
            size=int(len(real_values)*0.25)
        )
        synthetic_high = np.random.normal(
            loc=np.max(real_values)+3*std_dev,
            scale=std_dev/2, 
            size=int(len(real_values)*0.25)
        )
    else:  # Categorical features
        unique_vals = np.unique(real_values)
        synthetic_vals = np.random.choice(unique_vals, size=int(len(real_values)*0.5))
    
    negative_values = np.concatenate([
        passive_train[train_cutoff:, target_idx],
        passive_val[:, target_idx],
        passive_test[:, target_idx],
        synthetic_low if 'synthetic_low' in locals() else synthetic_vals,
        synthetic_high if 'synthetic_high' in locals() else np.array([])
    ])
    
    return (
        np.concatenate([real_values, negative_values]),
        np.concatenate([
            np.ones(len(real_values)),
            np.zeros(len(negative_values))
        ])
    )

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    active_raw = load_data(1, 'train')
    passive_train = load_data(2, 'train')
    
    # Initialize models
    active_model = BottomModel(input_dim=active_raw.shape[1]).to(device)
    with torch.no_grad():
        active_embs = active_model(torch.FloatTensor(active_raw).to(device)).cpu().numpy()

    # Analyze feature correlations before selecting target
    top_features = analyze_feature_correlations(active_embs, passive_train)
    print("\nTop 10 Correlated Features in Passive Client:")
    print("Index\t| Feature Type\t| Correlation Strength")
    print("---------------------------------------------")
    for idx, corr in top_features:
        feature_type = "Continuous" if len(np.unique(passive_train[:, idx])) > 10 else "Categorical"
        print(f"{idx}\t| {feature_type}\t| {corr:.4f}")
    
    # Select most correlated feature automatically
    target_feature_idx = top_features[0][0]
    print(f"\nAutomatically selected target feature: Column {target_feature_idx}")
    
    # Rest of the pipeline remains the same...
    passive_val = load_data(2, 'val')
    passive_test = load_data(2, 'test')
    passive_model = BottomModel(input_dim=passive_train.shape[1]).to(device)
    with torch.no_grad():
        passive_train_embs = passive_model(torch.FloatTensor(passive_train).to(device)).cpu().numpy()

    eval_values, y_true = create_evaluation_set(
        passive_train, passive_val, passive_test, target_feature_idx
    )

    attacker = FeatureInferenceAttack(target_feature_idx, active_embs.shape[1], device)
    attacker.train(active_embs, passive_train_embs, active_raw)

    y_pred, y_prob = [], []
    template_sample = passive_train[0].copy()
    
    for value in eval_values:
        sample = template_sample.copy()
        sample[target_feature_idx] = value
        
        with torch.no_grad():
            active_emb = active_model(torch.FloatTensor(active_raw[0:1]).to(device)).cpu().numpy()
            passive_emb = passive_model(torch.FloatTensor(sample[None,:]).to(device)).cpu().numpy()
        
        pred, prob = attacker.infer(active_emb, passive_emb)
        y_pred.append(pred)
        y_prob.append(prob)

    metrics = evaluate_attack(y_true, y_pred, y_prob)
    print("\n=== Enhanced Attack Results ===")
    print(f"Target Feature Index: {target_feature_idx}")
    print(f"Feature Type: {'Continuous' if len(np.unique(passive_train[:, target_feature_idx])) > 10 else 'Categorical'}")
    print(f"Real Samples: {sum(y_true)} | Fake Samples: {len(y_true)-sum(y_true)}")
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric.upper()}: {value:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

if __name__ == "__main__":
    main()
