from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

def get_accuracy(y_true, y_pred):
    """Computes basic accuracy score."""
    return accuracy_score(y_true, y_pred)

def get_macro_f1(y_true, y_pred):
    """Computes macro-averaged F1 score, useful for balanced/imbalanced class assessment."""
    return f1_score(y_true, y_pred, average='macro')

def get_confusion_matrix(y_true, y_pred):
    """Computes the confusion matrix."""
    return confusion_matrix(y_true, y_pred)

def evaluate_classification(y_true, y_pred):
    """
    Comprehensive evaluation helper for EuroSAT classification.
    
    Returns a dictionary containing:
    - Accuracy
    - Macro F1-score
    - Confusion Matrix
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'accuracy': get_accuracy(y_true, y_pred),
        'f1_macro': get_macro_f1(y_true, y_pred),
        'confusion_matrix': get_confusion_matrix(y_true, y_pred)
    }
    
    print(f"\n--- Classification Report ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    return metrics
