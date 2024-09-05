from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# List of available models
MODEL_LIST = {
    'logistic_regression': LogisticRegression,
    'decision_tree': DecisionTreeClassifier,
    'random_forest': RandomForestClassifier,
    'gradient_boosting': GradientBoostingClassifier,
    'svm': SVC,
    'naive_bayes': GaussianNB,
    'knn': KNeighborsClassifier,
    'xgboost': XGBClassifier,
    'kmeans': KMeans
}


def train_model(model_type, X_train, y_train, **kwargs):
    """
    Train a model based on the selected model type.
    
    Args:
    - model_type (str): Type of the model to train (e.g., 'logistic_regression', 'random_forest', etc.)
    - X_train (array-like): Training features
    - y_train (array-like): Training labels
    - kwargs: Additional arguments to customize model (e.g., hyperparameters)
    
    Returns:
    - model: Trained model
    """
    if model_type not in MODEL_LIST:
        raise ValueError(f"Model type '{model_type}' is not supported. Supported types: {list(MODEL_LIST.keys())}")
    
    model_class = MODEL_LIST[model_type]
    
    # Create an instance of the model with additional parameters if provided
    model = model_class(**kwargs)
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model


def train_neural_network(X_train, y_train, input_shape, output_shape, epochs=20, batch_size=32):
    """
    Train a simple neural network for classification using TensorFlow/Keras.

    Args:
    - X_train (array-like): Training features
    - y_train (array-like): Training labels
    - input_shape (int): Number of input features
    - output_shape (int): Number of output classes
    - epochs (int): Number of training epochs
    - batch_size (int): Training batch size
    
    Returns:
    - model: Trained Keras model
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='softmax' if output_shape > 1 else 'sigmoid')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy' if output_shape > 1 else 'binary_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model


def model_summary(model):
    """
    Generate a summary of the trained model, including feature importances or coefficients.
    
    Args:
    - model: Trained model
    
    Returns:
    - summary (dict): A dictionary summarizing the model (e.g., coefficients, feature importances)
    """
    summary = {}
    
    if hasattr(model, 'coef_'):
        # For linear models like Logistic Regression
        summary['coefficients'] = model.coef_.tolist()
    elif hasattr(model, 'feature_importances_'):
        # For tree-based models like Decision Trees, Random Forests
        summary['feature_importances'] = model.feature_importances_.tolist()
    else:
        summary['message'] = 'Model summary not available for this model type.'
    
    return summary


def hyperparameter_tuning(model_type, X_train, y_train, param_grid, cv=5):
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
    - model_type (str): Type of model (e.g., 'logistic_regression', 'random_forest', etc.)
    - X_train (array-like): Training features
    - y_train (array-like): Training labels
    - param_grid (dict): Dictionary containing parameter grid for tuning
    - cv (int): Number of cross-validation folds
    
    Returns:
    - best_model: The model with the best found hyperparameters
    - best_params: Best hyperparameters found during grid search
    """
    if model_type not in MODEL_LIST:
        raise ValueError(f"Model type '{model_type}' is not supported. Supported types: {list(MODEL_LIST.keys())}")
    
    model_class = MODEL_LIST[model_type]
    model = model_class()
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    return best_model, best_params


def ensemble_model(X_train, y_train):
    """
    Create an ensemble of models using a VotingClassifier.
    
    Args:
    - X_train (array-like): Training features
    - y_train (array-like): Training labels
    
    Returns:
    - ensemble (VotingClassifier): Trained VotingClassifier model
    """
    model1 = ('lr', LogisticRegression())
    model2 = ('rf', RandomForestClassifier())
    model3 = ('gb', GradientBoostingClassifier())
    
    ensemble = VotingClassifier(estimators=[model1, model2, model3], voting='soft')
    ensemble.fit(X_train, y_train)
    
    return ensemble


def cross_validate_model(model, X_train, y_train, cv=5):
    """
    Perform cross-validation on the given model.
    
    Args:
    - model: Model to evaluate
    - X_train (array-like): Training features
    - y_train (array-like): Training labels
    - cv (int): Number of cross-validation folds
    
    Returns:
    - scores (dict): Dictionary containing mean accuracy, precision, recall, and F1-score across folds
    """
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted')
    }
    
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    return {
        'mean_accuracy': np.mean(scores),
        'std_accuracy': np.std(scores),
        'min_accuracy': np.min(scores),
        'max_accuracy': np.max(scores)
    }


def stacking_ensemble(X_train, y_train):
    """
    Create a stacking ensemble of multiple models.
    
    Args:
    - X_train (array-like): Training features
    - y_train (array-like): Training labels
    
    Returns:
    - stacked_model (VotingClassifier): Trained stacking model
    """
    from sklearn.ensemble import StackingClassifier
    
    base_models = [
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier()),
        ('svc', SVC(probability=True))
    ]
    
    # Define meta-learner (Logistic Regression in this case)
    meta_model = LogisticRegression()
    
    stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
    stacked_model.fit(X_train, y_train)
    
    return stacked_model


def custom_preprocessing_pipeline(steps, X_train, y_train):
    """
    Apply a custom preprocessing pipeline defined by the user.
    
    Args:
    - steps (list of tuples): List of preprocessing steps (name, transformer)
    - X_train (array-like): Training features
    - y_train (array-like): Training labels
    
    Returns:
    - pipeline (Pipeline): Fitted preprocessing pipeline
    """
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    return pipeline


class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to demonstrate extending Scikit-learn's pipeline.
    """
    def __init__(self, param1=1, param2=2):
        self.param1 = param1
        self.param2 = param2

    def fit(self, X, y=None):
        # Fit method to compute any required stats from training data
        return self

    def transform(self, X):
        # Apply transformation (for example, multiplying by param1)
        return X * self.param1 + self.param2


def auto_ml(X_train, y_train):
    """
    Demonstrates an example of AutoML (automated machine learning) using hyperparameter search and ensembling.
    
    Args:
    - X_train (array-like): Training features
    - y_train (array-like): Training labels
    
    Returns:
    - best_model: Best-performing model found through AutoML
    """
    param_grid = {
        'random_forest': {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 20]},
        'svm': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    }
    
    best_models = {}
    
    for model_type in param_grid:
        best_model, best_params = hyperparameter_tuning(model_type, X_train, y_train, param_grid[model_type])
        best_models[model_type] = best_model
    
    # Assume we're selecting the best model based on accuracy from cross-validation
    best_model = max(best_models.values(), key=lambda model: cross_validate_model(model, X_train, y_train)['mean_accuracy'])
    
    return best_model
