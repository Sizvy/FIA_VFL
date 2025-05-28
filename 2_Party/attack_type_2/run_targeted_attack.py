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
            loc=np.mean(real_values) + 2 * std_dev,
            # loc=np.mean(real_values)
            scale=std_dev,
            # scale = std_dev/3,
            size=len(real_values)
        )
        # Filter out any synthetic values that accidentally match real ones
        synthetic = np.array([x for x in synthetic if not np.any(np.isclose(x, real_values, atol=std_dev/10))])
    else:  # Categorical features
        unique_vals = np.unique(real_values)
        all_possible_vals = set(range(np.max(unique_vals) + 5)) 
        synthetic_candidates = list(all_possible_vals - set(unique_vals))
        synthetic = np.random.choice(synthetic_candidates, size=len(real_values))
        # synthetic = np.random.choice(unique_vals, size=len(real_values))
        # synthetic = np.array([x for x in synthetic if x not in unique_vals])
    
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
            np.zeros(len(negative_values))
        ])
    )

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists('Saved_Models/best_vfl_model.pt'):
        print("Error: First train the utility model using without_dp.py")
        return

    checkpoint = torch.load('Saved_Models/best_vfl_model.pt')
    
    active_train = load_data(1, 'train')
    passive_train = load_data(2, 'train')
    passive_val = load_data(2, 'val')
    passive_test = load_data(2, 'test')

    active_model = BottomModel(input_dim=active_train.shape[1]).to(device)
    active_model.load_state_dict(checkpoint['client1_bottom'])
    
    passive_model = BottomModel(input_dim=passive_train.shape[1]).to(device)
    passive_model.load_state_dict(checkpoint['client2_bottom'])
    
    top_model = TopModel().to(device)
    with torch.no_grad():
        active_train_embs = active_model(torch.FloatTensor(active_train).to(device)).cpu().numpy()
    
        passive_train_embs = passive_model(torch.FloatTensor(passive_train).to(device)).cpu().numpy()
        
        passive_val_embs = passive_model(torch.FloatTensor(passive_val).to(device)).cpu().numpy()
        passive_test_embs = passive_model(torch.FloatTensor(passive_test).to(device)).cpu().numpy()

    num_features = passive_train.shape[1]

    for target_feature_idx in range(num_features):
        print("\n" + "="*40)
        print(f"Target feature selected: Column {target_feature_idx}")
        print(f"Feature type: {'Continuous' if len(np.unique(passive_train[:, target_feature_idx])) > 10 else 'Categorical'}")

        eval_values, y_true = create_evaluation_set(
            passive_train, passive_val, passive_test, target_feature_idx
        )

        real_combined = np.concatenate([active_train_embs, passive_train_embs], axis=1)
        real_labels = np.ones(len(real_combined))

        num_fake = len(passive_val_embs) + len(passive_test_embs)
        active_fake_embs = np.tile(active_train_embs[0], (num_fake, 1))
        fake_passive_embs = np.concatenate([passive_val_embs, passive_test_embs])
        fake_combined = np.concatenate([active_fake_embs, fake_passive_embs], axis=1)
        fake_labels = np.zeros(len(fake_combined))

        X_train = np.vstack([real_combined, fake_combined])
        y_train = np.concatenate([real_labels, fake_labels])

        attacker = FeatureInferenceAttack(
            target_feature_idx=target_feature_idx,
            emb_dim=active_train_embs.shape[1],
            device=device
        )
        attacker.train(X_train, y_train)

        all_passive_embs = np.concatenate([passive_train_embs, passive_val_embs, passive_test_embs])
        y_pred, y_prob = [], []

        num_samples = len(eval_values)
        for i in range(num_samples):
            active_idx = i % len(active_train_embs)
            passive_idx = i % len(all_passive_embs)
            active_emb = active_train_embs[active_idx]
            passive_emb = all_passive_embs[passive_idx]
            if isinstance(active_emb, torch.Tensor):
                active_emb = active_emb.cpu().numpy()
            if isinstance(passive_emb, torch.Tensor):
                passive_emb = passive_emb.cpu().numpy()
            pred, prob = attacker.infer(active_emb, passive_emb)
            y_pred.append(pred)
            y_prob.append(prob)

        metrics = evaluate_attack(y_true, y_pred, y_prob)
        print("=== Attack Results ===")
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
