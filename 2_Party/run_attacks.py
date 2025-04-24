import numpy as np
import torch
from models.averageBottom import BottomModel
from attacks.embedding_discriminator import EmbeddingDiscriminatorAttack

def load_embeddings():
    client1_train = np.load('splitted_data/client_1_train.npy')
    client2_train = np.load('splitted_data/client_2_train.npy')
    
    active_bottom = BottomModel(input_dim=client1_train.shape[1])
    passive_bottom = BottomModel(input_dim=client2_train.shape[1])
    
    checkpoint = torch.load('Saved_Models/best_vfl_model.pt')
    active_bottom.load_state_dict(checkpoint['client1_bottom'], strict=False)
    passive_bottom.load_state_dict(checkpoint['client2_bottom'], strict=False)
    
    with torch.no_grad():
        active_embs = active_bottom(torch.FloatTensor(client1_train)).cpu().numpy()
        passive_embs = passive_bottom(torch.FloatTensor(client2_train)).cpu().numpy()
    
    return active_embs, passive_embs

def evaluate_attack(n_trials=10):
    active_embs, passive_embs = load_embeddings()
    emb_dim = active_embs.shape[1]
    
    all_metrics = []
    
    for trial in range(n_trials):
        print(f"\n=== Trial {trial+1}/{n_trials} ===")
        torch.manual_seed(trial)
        np.random.seed(trial)
        
        attacker = EmbeddingDiscriminatorAttack(
            active_dim=active_embs.shape[1],
            passive_dim=passive_embs.shape[1],
            emb_dim=emb_dim
        )
        
        metrics = attacker.train_model(active_embs, passive_embs)
        all_metrics.append(metrics)
    
    print("\n=== Final Results ===")
    for metric in ['auc', 'accuracy', 'f1', 'precision', 'recall']:
        values = [m[metric] for m in all_metrics]
        print(f"{metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")

if __name__ == "__main__":
    evaluate_attack(n_trials=10)
