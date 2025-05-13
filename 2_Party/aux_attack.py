import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.averageBottom import BottomModel
from models.simpleTop import TopModel
from attacks.aux_attack_train import FeatureInferenceAttack
from utils.evaluate import evaluate_attack

def load_data(client_num, split='train'):
    return np.load(f'splitted_data/client_{client_num}_{split}.npy')

def create_evaluation_set(passive_train, passive_val, passive_test, target_idx):
    real_values = passive_train[:, target_idx]
    
    if len(np.unique(real_values)) > 10:  # Continuous
        std_dev = np.std(real_values)
        synthetic = np.random.normal(
            loc=np.mean(real_values) + 2 * std_dev,
            scale=std_dev,
            size=len(real_values))
        synthetic = np.array([x for x in synthetic if not np.any(np.isclose(x, real_values, atol=std_dev/10))])
    else:  # Categorical
        unique_vals = np.unique(real_values)
        all_possible_vals = set(range(np.max(unique_vals) + 5)) 
        synthetic_candidates = list(all_possible_vals - set(unique_vals))
        synthetic = np.random.choice(synthetic_candidates, size=len(real_values))
    
    negative_values = np.concatenate([
        passive_val[:, target_idx],
        passive_test[:, target_idx],
        synthetic
    ])
    
    return (
        np.concatenate([real_values, negative_values]),
        np.concatenate([np.ones(len(real_values)), np.zeros(len(negative_values))])
    )

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load models and data
    checkpoint = torch.load('Saved_Models/best_vfl_model.pt')
    active_train = load_data(1, 'train')
    passive_train = load_data(2, 'train')
    passive_val = load_data(2, 'val')
    passive_test = load_data(2, 'test')

    # Initialize models
    active_model = BottomModel(input_dim=active_train.shape[1]).to(device)
    passive_model = BottomModel(input_dim=passive_train.shape[1]).to(device)
    active_model.load_state_dict(checkpoint['client1_bottom'])
    passive_model.load_state_dict(checkpoint['client2_bottom'])
    
    # Get embeddings
    with torch.no_grad():
        active_train_embs = active_model(torch.FloatTensor(active_train).to(device)).cpu().numpy()
        passive_train_embs = passive_model(torch.FloatTensor(passive_train).to(device)).cpu().numpy()
        passive_val_embs = passive_model(torch.FloatTensor(passive_val).to(device)).cpu().numpy()
        passive_test_embs = passive_model(torch.FloatTensor(passive_test).to(device)).cpu().numpy()

    target_feature_idx = 0  # Change as needed
    
    # Create known samples (1% of data)
    n_known = int(0.01 * len(passive_train))
    known_indices = np.random.choice(len(passive_train), n_known, replace=False)
    known_samples = {
        passive_train[i, target_feature_idx]: (
            torch.FloatTensor(active_train_embs[i]),
            torch.FloatTensor(passive_train_embs[i])
        )
        for i in known_indices
    }

    # Prepare training data
    real_combined = np.concatenate([active_train_embs, passive_train_embs], axis=1)
    real_labels = np.ones(len(real_combined))
    
    num_fake = len(passive_val_embs) + len(passive_test_embs)
    active_fake_embs = np.tile(active_train_embs[0], (num_fake, 1))
    fake_passive_embs = np.concatenate([passive_val_embs, passive_test_embs])
    fake_combined = np.concatenate([active_fake_embs, fake_passive_embs], axis=1)
    fake_labels = np.zeros(len(fake_combined))
    
    X_train = np.vstack([real_combined, fake_combined])
    y_train = np.concatenate([real_labels, fake_labels])
    feature_values = np.concatenate([
        passive_train[:, target_feature_idx],
        np.zeros(len(fake_combined))  # Dummy values for fake samples
    ])

    # Initialize and train attacker
    attacker = FeatureInferenceAttack(
        target_feature_idx=target_feature_idx,
        emb_dim=active_train_embs.shape[1],
        device=device,
        known_samples=known_samples
    )
    attacker.train(X_train, y_train, feature_values)

    # Evaluation
    eval_values, y_true = create_evaluation_set(passive_train, passive_val, passive_test, target_feature_idx)
    all_passive_embs = np.concatenate([passive_train_embs, passive_val_embs, passive_test_embs])
    
    y_pred, y_prob, y_feature_pred = [], [], []
    for i in range(len(eval_values)):
        active_idx = i % len(active_train_embs)
        passive_idx = i % len(all_passive_embs)
        
        is_real, pred_value = attacker.infer(
            active_train_embs[active_idx],
            all_passive_embs[passive_idx]
        )
        y_pred.append(is_real)
        y_prob.append(1.0 if is_real else 0.0)  # Probability of being real
        y_feature_pred.append(pred_value)

    # Calculate metrics
    metrics = evaluate_attack(y_true, y_pred, y_prob)
    
    # Feature prediction metrics
    if len(np.unique(passive_train[:, target_feature_idx])) > 10:  # Continuous
        mae = np.mean(np.abs(np.array(y_feature_pred)[y_true==1] - eval_values[y_true==1]))
        print(f"\nFeature MAE: {mae:.4f}")
    else:  # Categorical
        accuracy = np.mean(np.round(y_feature_pred)[y_true==1] == eval_values[y_true==1])
        print(f"\nFeature Accuracy: {accuracy:.4f}")

    print("\n=== Attack Results ===")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric.upper()}: {value:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

if __name__ == "__main__":
    main()
