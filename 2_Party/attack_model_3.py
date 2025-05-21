import torch
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

def train_gbdt_attack_model():
    # Load and prepare data
    attack_train_data = np.load('attack_model_data/attack_train_data.npy')
    attack_test_data = np.load('attack_model_data/attack_test_data.npy')
    all_data = np.concatenate([attack_train_data, attack_test_data])
    
    # Create ratio features between prediction vector components
    preds = all_data[:, :11]
    ratios = np.zeros((preds.shape[0], 55))  # 11 choose 2 ratios
    idx = 0
    for i in range(11):
        for j in range(i+1, 11):
            ratios[:, idx] = preds[:, i] / (preds[:, j] + 1e-8)
            idx += 1
    
    # Combine with original features
    X = np.hstack([all_data[:, :11], ratios, all_data[:, 11:12]])
    y = all_data[:, -1]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train Gradient Boosted Trees
    model = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=5,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    
    print(f"\nTest AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Save model
    joblib.dump(model, 'Saved_Models/gbdt_attack_model.pkl')
    print("Model saved")

if __name__ == "__main__":
    train_gbdt_attack_model()
