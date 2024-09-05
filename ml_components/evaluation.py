import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize

def evaluate_model(model, X_test, y_test, average='weighted', multi_class=False):
    """
    Evaluate a machine learning model using various metrics such as accuracy, precision, recall, F1 score, etc.
    
    Args:
    - model: Trained model to evaluate
    - X_test: Test data features
    - y_test: Test data labels
    - average: Type of averaging to use for multiclass (default: 'weighted')
    - multi_class: Boolean to indicate if it's a multiclass problem (default: False)
    
    Returns:
    - metrics: A dictionary of evaluation metrics
    """
    # Predict the labels for test set
    y_pred = model.predict(X_test)
    
    # Calculate the core metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average=average),
        'recall': recall_score(y_test, y_pred, average=average),
        'f1_score': f1_score(y_test, y_pred, average=average),
    }
    
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        if multi_class:
            # For multiclass, calculate the ROC AUC score for each class
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            metrics['roc_auc'] = roc_auc_score(y_test_bin, y_proba, average=average, multi_class='ovo')
        else:
            # For binary classification, calculate the ROC AUC score
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
    
    # Generate a classification report (for detailed output)
    metrics['classification_report'] = classification_report(y_test, y_pred)
    
    return metrics


def plot_confusion_matrix(model, X_test, y_test, labels=None, normalize=False):
    """
    Plot a confusion matrix for a given model and test set.
    
    Args:
    - model: Trained model
    - X_test: Test data features
    - y_test: Test data labels
    - labels: List of class labels for the confusion matrix axes (optional)
    - normalize: Whether to normalize the confusion matrix (default: False)
    
    Returns:
    - None (displays the confusion matrix)
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_roc_curve(model, X_test, y_test, multi_class=False, labels=None):
    """
    Plot the ROC curve for binary or multiclass classification.
    
    Args:
    - model: Trained model
    - X_test: Test data features
    - y_test: Test data labels
    - multi_class: Boolean to indicate if it's a multiclass problem (default: False)
    - labels: List of class labels for the ROC curve plot (optional, only used in multiclass)
    
    Returns:
    - None (displays the ROC curve)
    """
    if not hasattr(model, 'predict_proba'):
        raise ValueError('Model does not support probability predictions required for ROC curve.')
    
    if multi_class:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        y_proba = model.predict_proba(X_test)
        n_classes = y_test_bin.shape[1]
        
        plt.figure(figsize=(10, 7))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            plt.plot(fpr, tpr, lw=2, label=f'Class {i if labels is None else labels[i]} (area = {roc_auc_score(y_test_bin[:, i], y_proba[:, i]):.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Multiclass Classification')
        plt.legend(loc="lower right")
        plt.show()
    else:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(y_test, y_proba):.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Binary Classification')
        plt.legend(loc="lower right")
        plt.show()


def plot_feature_importance(model, feature_names):
    """
    Plot the feature importance for models that support this attribute (e.g., tree-based models).
    
    Args:
    - model: Trained model
    - feature_names: List of feature names
    
    Returns:
    - None (displays the feature importance plot)
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError('Model does not have feature importances.')

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 7))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()


def evaluate_and_plot_all(model, X_test, y_test, feature_names=None, labels=None, multi_class=False):
    """
    Evaluate a model and generate all available evaluation plots including confusion matrix, ROC curve, and feature importance.
    
    Args:
    - model: Trained model
    - X_test: Test data features
    - y_test: Test data labels
    - feature_names: List of feature names (optional, used for feature importance)
    - labels: List of class labels (optional, used for confusion matrix and ROC curve)
    - multi_class: Boolean indicating whether it's a multiclass classification problem (default: False)
    
    Returns:
    - metrics: A dictionary of evaluation metrics
    """
    # Evaluate the model and print metrics
    metrics = evaluate_model(model, X_test, y_test, multi_class=multi_class)
    print("Model Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"F1-Score: {metrics['f1_score']:.2f}")
    
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.2f}")
    
    print("\nClassification Report:\n", metrics['classification_report'])
    
    # Plot the confusion matrix
    plot_confusion_matrix(model, X_test, y_test, labels=labels)
    
    # Plot ROC curve
    plot_roc_curve(model, X_test, y_test, multi_class=multi_class, labels=labels)
    
    # Plot feature importance (if applicable)
    if feature_names is not None and hasattr(model, 'feature_importances_'):
        plot_feature_importance(model, feature_names)
    
    return metrics
