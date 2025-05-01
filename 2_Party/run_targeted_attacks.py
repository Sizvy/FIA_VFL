import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.averageBottom import BottomModel
from models.simpleTop import TopModel
from attacks.feature_inference import FeatureInferenceAttack
from utils.evaluate import evaluate_attack

def load_data(client_num, split='train'):
    """Load numpy array for specified client and split"""
    return np.load(f'splitted_data/client_{client_num}_{split}.npy')

def create_evaluation_set(passive_train, passive_val, passive_test, target_idx):
    """Create evaluation set with perfect label consistency"""
    # ALL training data is considered "real"
    real_values = passive_train[:, target_idx]
    
    # Generate synthetic negatives that are close to real values but not present
    if len(np.unique(real_values)) > 10:  # Continuous features
        std_dev = np.std(real_values)
        synthetic = np.random.normal(
            loc=np.mean(real_values),
            scale=std_dev/3,
            size=len(real_values)
        )
        # Filter out any synthetic values that accidentally match real ones
        synthetic = np.array([x for x in synthetic if not np.any(np.isclose(x, real_values, atol=std_dev/10))])
    else:  # Categorical features
        unique_vals = np.unique(real_values)
        synthetic = np.random.choice(unique_vals, size=len(real_values))
        synthetic = np.array([x for x in synthetic if x not in unique_vals])
    
    # Combine validation, test and synthetic samples as negatives
    negative_values = np.concatenate([
        passive_val[:, target_idx],
        passive_test[:, target_idx],
        synthetic
    ])
    
    return (
        np.concatenate([real_values, negative_values]),
        np.concatenate([
            np.ones(len(real_values)),  # All training data is real
            np.zeros(len(negative_values))  # Everything else is fake
        ])
    )

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Verify utility model exists
    if not os.path.exists('Saved_Models/best_vfl_model.pt'):
        print("Error: First train the utility model using without_dp.py")
        return

    # 2. Load trained models from utility training
    checkpoint = torch.load('Saved_Models/best_vfl_model.pt')
    
    # Load data - active client can only access their own raw data
    active_train = load_data(1, 'train')
    passive_train = load_data(2, 'train')
    passive_val = load_data(2, 'val')
    passive_test = load_data(2, 'test')

    # Initialize models WITH TRAINED WEIGHTS
    active_model = BottomModel(input_dim=active_train.shape[1]).to(device)
    active_model.load_state_dict(checkpoint['client1_bottom'])
    
    passive_model = BottomModel(input_dim=passive_train.shape[1]).to(device)
    passive_model.load_state_dict(checkpoint['client2_bottom'])
    
    top_model = TopModel().to(device)
    top_model.load_state_dict(checkpoint['top_model'])

    # Generate embeddings using trained models
    with torch.no_grad():
        # Active client can generate embeddings for their data
        active_train_embs = active_model(torch.FloatTensor(active_train).to(device)).cpu().numpy()
        
        # Passive embeddings available during training
        passive_train_embs = passive_model(torch.FloatTensor(passive_train).to(device)).cpu().numpy()
        
        # For evaluation only
        passive_val_embs = passive_model(torch.FloatTensor(passive_val).to(device)).cpu().numpy()
        passive_test_embs = passive_model(torch.FloatTensor(passive_test).to(device)).cpu().numpy()

    # Manually set target feature index
    target_feature_idx = 0  # Change this to the column index you want to attack
    print(f"\nTarget feature selected: Column {target_feature_idx}")
    print(f"Feature type: {'Continuous' if len(np.unique(passive_train[:, target_feature_idx])) > 10 else 'Categorical'}")

    # Create evaluation set
    eval_values, y_true = create_evaluation_set(
        passive_train, passive_val, passive_test, target_feature_idx
    )

    # Initialize attack model with correct embedding dimension
    attacker = FeatureInferenceAttack(
        target_feature_idx=target_feature_idx,
        emb_dim=active_train_embs.shape[1],  # Dimension of single party's embeddings
        device=device
    )
    
    # Train attack model using only what active client has access to:
    # - Active raw features (for labels)
    # - Active embeddings (from their bottom model)
    # - Passive embeddings (from joint training)
    attacker.train(
        active_embs=active_train_embs,
        passive_embs=passive_train_embs,
        active_raw=active_train
    )

    # Prepare evaluation data
    all_passive_embs = np.concatenate([passive_train_embs, passive_val_embs, passive_test_embs])
    
    # Run inference
    y_pred, y_prob = [], []
    num_samples = len(eval_values)
    for i in range(num_samples):
        passive_idx = i % len(all_passive_embs)
        # Get embeddings with proper shape
        active_emb = active_train_embs[0]  # Shape: [embedding_dim]
        passive_emb = all_passive_embs[passive_idx]  # Shape: [embedding_dim]
    
        # Ensure they're numpy arrays
        if isinstance(active_emb, torch.Tensor):
            active_emb = active_emb.cpu().numpy()
        if isinstance(passive_emb, torch.Tensor):
            passive_emb = passive_emb.cpu().numpy()
    
        # Make prediction
        pred, prob = attacker.infer(active_emb, passive_emb)
        y_pred.append(pred)
        y_prob.append(prob)

    # Evaluate results
    metrics = evaluate_attack(y_true, y_pred, y_prob)
    print("\n=== Attack Results ===")
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
