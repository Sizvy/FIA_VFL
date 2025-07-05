import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_distance_classifier(distances_dict, test_size=0.2, model_type='logistic', features=['euclidean', 'manhattan', 'cosine']):
    """Train classifier with specified distance features"""
    # Validate feature selection
    valid_features = ['euclidean', 'manhattan', 'cosine']
    if not all(f in valid_features for f in features):
        raise ValueError(f"Invalid features. Choose from {valid_features}")
    
    # Prepare selected features
    X_parts = []
    for feat in features:
        X_parts.append(np.concatenate([
            distances_dict['intra'][feat], 
            distances_dict['inter'][feat]
        ]))
    
    X = np.column_stack(X_parts)
    y = np.concatenate([
        np.zeros(len(distances_dict['intra']['euclidean'])),
        np.ones(len(distances_dict['inter']['euclidean']))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Initialize model
    if model_type == 'logistic':
        clf = LogisticRegression()
    elif model_type == 'svm':
        clf = SVC(kernel='rbf', probability=True)
    elif model_type == 'random_forest':
        clf = RandomForestClassifier()
    else:
        raise ValueError("Invalid model_type. Choose from 'logistic', 'svm', or 'random_forest'")
    
    # Train and evaluate
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else clf.decision_function(X_test)
    
    # Generate reports
    report = classification_report(y_test, y_pred, target_names=['Intra', 'Inter'])
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Plot decision boundary
    plot_decision_boundary(clf, X, y, model_type)
    
    return clf, report, roc_auc

def plot_decision_boundary(clf, X, y, model_name):
    """Visualizes how the classifier separates the two distance types"""
    plt.figure(figsize=(10, 6))
    
    # Plot raw distances
    sns.kdeplot(X[y==0].ravel(), label='Intra-Sample', fill=True)
    sns.kdeplot(X[y==1].ravel(), label='Inter-Sample', fill=True)
    
    # Plot decision threshold
    if hasattr(clf, 'predict_proba'):
        threshold = np.mean(clf.predict_proba(X)[:, 1])
    else:
        threshold = 0
    plt.axvline(x=threshold, color='red', linestyle='--', 
               label=f'Decision Boundary ({threshold:.2f})')
    
    plt.title(f'Distance Classification ({model_name})\nDecision Boundary Visualization')
    plt.xlabel('Embedding Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'../shadow_model_data/{model_name}_decision_boundary.png')
    plt.show()

if __name__ == "__main__":
    # Example usage (you'll call this from your main script)
    print("This module is meant to be imported, not run directly")
