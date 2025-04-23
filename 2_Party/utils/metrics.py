from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                           recall_score, f1_score, confusion_matrix,
                           average_precision_score, classification_report)

def calculate_metrics(preds, labels):
    preds_binary = (preds > 0.5).astype(int)
    return {
        'auc': roc_auc_score(labels, preds),
        'ap': average_precision_score(labels, preds),
        'accuracy': accuracy_score(labels, preds_binary),
        'precision': precision_score(labels, preds_binary),
        'recall': recall_score(labels, preds_binary),
        'f1': f1_score(labels, preds_binary),
        'confusion_matrix': confusion_matrix(labels, preds_binary),
        'classification_report': classification_report(labels, preds_binary, 
                                                     target_names=['Shuffled', 'Real'])
    }
