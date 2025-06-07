import numpy as np
import torch
from models.averageBottom import BottomModel
from attacks.embedding_discriminator import EmbeddingDiscriminatorAttack
import os

def train_and_save_attack_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    client1_train = np.load('splitted_data/client_1_train.npy')
    client2_train = np.load('splitted_data/client_2_train.npy')

    # Initialize models
    active_bottom = BottomModel(input_dim=client1_train.shape[1]).to(device)
    passive_bottom = BottomModel(input_dim=client2_train.shape[1]).to(device)

    # Load pretrained weights
    checkpoint = torch.load('Saved_Models/best_vfl_model.pt')
    active_bottom.load_state_dict(checkpoint['client1_bottom'], strict=False)
    passive_bottom.load_state_dict(checkpoint['client2_bottom'], strict=False)

    active_bottom.eval()
    passive_bottom.eval()

    # Get all embeddings
    with torch.no_grad():
        active_embs = active_bottom(torch.FloatTensor(client1_train).to(device)).cpu().numpy()
        passive_embs = passive_bottom(torch.FloatTensor(client2_train).to(device)).cpu().numpy()

    # Determine embedding dimension
    emb_dim = active_embs.shape[1]

    # Initialize and train attack model
    attacker = EmbeddingDiscriminatorAttack(
        active_dim=client1_train.shape[1],
        passive_dim=client2_train.shape[1],
        emb_dim=emb_dim
    )
    
    # Train the model
    attacker.train_model(active_embs, passive_embs, epochs=100)
    
    # Create directory if it doesn't exist
    os.makedirs('Saved_Models/attack_models', exist_ok=True)
    
    # Save the attack model
    torch.save({
        'discriminator_state': attacker.discriminator.state_dict(),
        'emb_dim': emb_dim,
        'active_dim': client1_train.shape[1],
        'passive_dim': client2_train.shape[1]
    }, 'Saved_Models/embedding_discriminator.pt')
    
    print("Attack model trained and saved successfully.")

if __name__ == "__main__":
    train_and_save_attack_model()
