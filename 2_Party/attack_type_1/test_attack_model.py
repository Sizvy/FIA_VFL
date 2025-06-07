import numpy as np
import torch
from models.averageBottom import BottomModel
from attacks.embedding_discriminator import EmbeddingDiscriminatorAttack

def load_and_test_attack_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data (should be different from training data)
    client1_test = np.load('splitted_data/client_1_train.npy')
    client2_test = np.load('splitted_data/client_2_train.npy')

    # Initialize bottom models
    active_bottom = BottomModel(input_dim=client1_test.shape[1]).to(device)
    passive_bottom = BottomModel(input_dim=client2_test.shape[1]).to(device)

    # Load pretrained weights for bottom models
    checkpoint = torch.load('Saved_Models/best_vfl_model.pt')
    active_bottom.load_state_dict(checkpoint['client1_bottom'], strict=False)
    passive_bottom.load_state_dict(checkpoint['client2_bottom'], strict=False)

    active_bottom.eval()
    passive_bottom.eval()

    # Compute test embeddings
    with torch.no_grad():
        active_embs_test = active_bottom(torch.FloatTensor(client1_test).to(device)).cpu().numpy()
        passive_embs_test = passive_bottom(torch.FloatTensor(client2_test).to(device)).cpu().numpy()

    # Load the attack model
    attack_checkpoint = torch.load('Saved_Models/embedding_discriminator.pt')
    
    # Initialize attack model
    attacker = EmbeddingDiscriminatorAttack(
        active_dim=attack_checkpoint['active_dim'],
        passive_dim=attack_checkpoint['passive_dim'],
        emb_dim=attack_checkpoint['emb_dim']
    )
    
    # Load the trained discriminator weights
    attacker.discriminator.load_state_dict(attack_checkpoint['discriminator_state'])
    attacker.discriminator.to(device)
    
    # Evaluate on test data
    test_metrics = attacker.validate_final(active_embs_test, passive_embs_test)
    
    print("\n=== Attack Model Test Results ===")
    print(f"AUC: {test_metrics['auc']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")

if __name__ == "__main__":
    load_and_test_attack_model()
