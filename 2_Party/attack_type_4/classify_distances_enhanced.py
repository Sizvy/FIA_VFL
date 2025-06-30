import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_distance_classifier(intra_distances, inter_distances, test_size=0.2, model_type='logistic'):
    """
    Enhanced classifier with:
    1. Multiple distance features (Euclidean + Manhattan)
    2. Cost-sensitive learning
    """
    # --- Improvement 1: Feature Engineering ---
    # Prepare multiple distance metrics as features
    X = np.column_stack([
        np.concatenate([intra_distances['euclidean'], inter_distances['euclidean']),
        np.concatenate([intra_distances['manhattan'], inter_distances['manhattan'])
    ])
    
    y = np.concatenate([np.zeros(len(intra_distances['euclidean'])),  # 0 for intra
                       np.ones(len(inter_distances['euclidean']))])   # 1 for inter
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # --- Improvement 2: Cost-Sensitive Learning ---
    # Initialize model with class weights
    model_params = {
        'class_weight': {'Intra': 1, 'Inter': 2}  # 2x penalty for misclassifying Inter
    }
    
    if model_type == 'logistic':
        clf = LogisticRegression(**model_params)
    elif model_type == 'svm':
        clf = SVC(kernel='rbf', probability=True, **model_params)
    elif model_type == 'random_forest':
        clf = RandomForestClassifier(**model_params)
    else:
        raise ValueError("Invalid model_type. Choose from 'logistic', 'svm', or 'random_forest'")
    
    # --- Original Training Pipeline ---
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else clf.decision_function(X_test)
    
    # Generate reports
    report = classification_report(y_test, y_pred, target_names=['Intra', 'Inter'])
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Plot decision boundary (now 2D)
    plot_decision_boundary(clf, X, y, model_type)
    
    return clf, report, roc_auc

def plot_decision_boundary(clf, X, y, model_name):
    """Enhanced 2D visualization for multiple features"""
    plt.figure(figsize=(12, 6))
    
    # Create grid for decision surface
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict over grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision surface
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    
    # Plot original data points
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, 
                   palette={0:'blue', 1:'red'}, 
                   style=y, markers={'Intra':'o', 'Inter':'s'},
                   alpha=0.5)
    
    plt.title(f'2D Decision Boundary ({model_name})')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Manhattan Distance')
    plt.legend(title='Sample Type')
    plt.savefig(f'../shadow_model_data/{model_name}_2d_decision_boundary.png')
    plt.show()

if __name__ == "__main__":
    print("This module is meant to be imported, not run directly")
