import numpy as np
import torch
from models.averageBottom import BottomModel
from attacks.feature_inference import FeatureInferenceAttack
from utils.evaluate import evaluate_attack

def load_data(client_num, split='train'):
    return np.load(f'splitted_data/client_{client_num}_{split}.npy')

def create_evaluation_set(passive_train, passive_val, passive_test, target_idx, contamination=0.3):
    real_values = passive_train[:, target_idx]
    val_test_values = np.concatenate([passive_val[:, target_idx], passive_test[:, target_idx]])
    synthetic_values = np.random.uniform(
        low=np.min(real_values)*1.1,
        high=np.max(real_values)*1.1,
        size=int(len(real_values)*contamination)
    )
    return (
        np.concatenate([real_values, val_test_values, synthetic_values]),
        np.concatenate([
            np.ones(len(real_values)),
            np.zeros(len(val_test_values) + len(synthetic_values))
        ])
    )

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_feature_idx = 10
    
    active_raw = load_data(1, 'train')
    active_model = BottomModel(input_dim=active_raw.shape[1]).to(device)
    active_embs = active_model(torch.FloatTensor(active_raw).to(device)).cpu().detach().numpy()

    passive_train = load_data(2, 'train')
    passive_val = load_data(2, 'val')
    passive_test = load_data(2, 'test')
    passive_model = BottomModel(input_dim=passive_train.shape[1]).to(device)

    eval_values, y_true = create_evaluation_set(
        passive_train, passive_val, passive_test, target_feature_idx
    )

    passive_train_embs = passive_model(torch.FloatTensor(passive_train).to(device)).cpu().detach().numpy()
    attacker = FeatureInferenceAttack(target_feature_idx, active_embs.shape[1], device)
    attacker.train(active_embs, passive_train_embs, active_raw)

    y_pred, y_prob = [], []
    for value in eval_values:
        sample = passive_train[0].copy()
        sample[target_feature_idx] = value
        with torch.no_grad():
            active_emb = active_model(torch.FloatTensor(active_raw[0:1]).to(device)).cpu().numpy()
            passive_emb = passive_model(torch.FloatTensor(sample[None,:]).to(device)).cpu().numpy()
        pred, prob = attacker.infer(active_emb, passive_emb)
        y_pred.append(pred)
        y_prob.append(prob)

    metrics = evaluate_attack(y_true, y_pred, y_prob)
    print("\n=== Evaluation Results ===")
    print(f"Real Samples: {len(passive_train)} | Fake Samples: {len(y_true)-len(passive_train)}")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric.upper()}: {value:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])

if __name__ == "__main__":
    main()
