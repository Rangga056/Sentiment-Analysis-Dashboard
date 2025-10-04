# ============================================================================
# MODEL TRAINING UTILITIES
# Machine learning model training with hyperparameter optimization
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')


def prepare_data(df, text_column, test_size=0.2, random_state=42):
    """
    Prepare data for model training
    
    Args:
        df: DataFrame with text and sentiment columns
        text_column: Name of text column
        test_size: Proportion of test set
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test, sentiment_map
    """
    # Map sentiments to numeric labels
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    X = df[text_column].values
    y = df['sentiment_category'].map(sentiment_map).values
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test, sentiment_map


def get_param_grid(model_name, analysis_mode='Quick Mode'):
    """
    Get hyperparameter grid for model
    
    Args:
        model_name: Name of model (SVM, KNN)
        analysis_mode: 'Quick Mode' or 'Full Analysis'
    
    Returns:
        Dictionary of parameters
    """
    if analysis_mode == 'Quick Mode':
        param_grids = {
            'SVM': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale'],
                'vectorizer__max_features': [3000],
                'vectorizer__ngram_range': [(1, 2)]
            },
            'KNN': {
                'classifier__n_neighbors': [3, 5, 7],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['euclidean'],
                'vectorizer__max_features': [3000],
                'vectorizer__ngram_range': [(1, 2)]
            }
        }
    else:  # Full Analysis
        param_grids = {
            'SVM': {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__kernel': ['linear', 'rbf', 'poly'],
                'classifier__gamma': ['scale', 'auto'],
                'vectorizer__max_features': [3000, 5000],
                'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)]
            },
            'KNN': {
                'classifier__n_neighbors': [3, 5, 7, 9, 11],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['euclidean', 'manhattan', 'cosine'],
                'vectorizer__max_features': [3000, 5000],
                'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)]
            }
        }
    
    return param_grids.get(model_name, {})


def train_single_model(X_train, X_test, y_train, y_test, model_name,
                      analysis_mode='Quick Mode', cv_folds=5, smote_k=5):
    """
    Train a single model with hyperparameter optimization
    
    Args:
        X_train, X_test, y_train, y_test: Train/test splits
        model_name: Name of model to train
        analysis_mode: Quick or Full analysis
        cv_folds: Number of CV folds
        smote_k: K neighbors for SMOTE
    
    Returns:
        Dictionary with model and metrics
    """
    # Get base model
    if model_name == 'SVM':
        base_model = SVC(probability=True, random_state=42)
    elif model_name == 'KNN':
        base_model = KNeighborsClassifier()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create pipeline with SMOTE
    pipeline = ImbPipeline([
        ('vectorizer', TfidfVectorizer()),
        ('smote', SMOTE(k_neighbors=smote_k, random_state=42)),
        ('classifier', base_model)
    ])
    
    # Get parameter grid
    param_grid = get_param_grid(model_name, analysis_mode)
    
    # GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_folds,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, 
                                                       target_names=['negative', 'neutral', 'positive'],
                                                       output_dict=True, zero_division=0),
        'predictions': y_pred
    }
    
    # ROC curve data for multiclass
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_data = {}
    
    for i, class_name in enumerate(['negative', 'neutral', 'positive']):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        roc_data[class_name] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': roc_auc
        }
    
    metrics['roc_data'] = roc_data
    
    return {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'metrics': metrics
    }


def train_all_models(X_train, X_test, y_train, y_test, model_names,
                    analysis_mode='Quick Mode', cv_folds=5, smote_k=5,
                    progress_callback=None):
    """
    Train multiple models
    
    Args:
        X_train, X_test, y_train, y_test: Train/test splits
        model_names: List of model names to train
        analysis_mode: Quick or Full analysis
        cv_folds: Number of CV folds
        smote_k: K neighbors for SMOTE
        progress_callback: Function to update progress
    
    Returns:
        Dictionary with results for each model
    """
    results = {}
    total = len(model_names)
    
    for i, model_name in enumerate(model_names):
        if progress_callback:
            progress_callback(i, total, f"Training {model_name}...")
        
        result = train_single_model(
            X_train, X_test, y_train, y_test,
            model_name, analysis_mode, cv_folds, smote_k
        )
        
        results[model_name] = result
        
        if progress_callback:
            progress_callback(i + 1, total, f"Completed {model_name}")
    
    return results


def compare_models(results):
    """
    Compare multiple trained models
    
    Args:
        results: Dictionary of model results
    
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []
    
    for model_name, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Macro': metrics['f1_macro'],
            'F1-Weighted': metrics['f1_weighted']
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    return df


def get_feature_importance(model, top_n=20):
    """
    Extract feature importance from trained model
    
    Args:
        model: Trained pipeline model
        top_n: Number of top features to return
    
    Returns:
        Dictionary with feature importance per class
    """
    try:
        # Get vectorizer and classifier from pipeline
        vectorizer = model.named_steps['vectorizer']
        classifier = model.named_steps['classifier']
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # For SVM with linear kernel
        if hasattr(classifier, 'coef_'):
            importance = {}
            
            for i, class_name in enumerate(['negative', 'neutral', 'positive']):
                if len(classifier.coef_) > i:
                    # Get coefficients for this class
                    coef = classifier.coef_[i]
                    
                    # Get top positive and negative features
                    top_positive_idx = np.argsort(coef)[-top_n:][::-1]
                    top_negative_idx = np.argsort(coef)[:top_n]
                    
                    importance[class_name] = {
                        'positive': [(feature_names[idx], coef[idx]) 
                                   for idx in top_positive_idx],
                        'negative': [(feature_names[idx], coef[idx]) 
                                   for idx in top_negative_idx]
                    }
            
            return importance
        else:
            return None
    
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        return None


def get_misclassified_samples(X_test, y_test, y_pred, sentiment_map, max_samples=100):
    """
    Get misclassified samples for analysis
    
    Args:
        X_test: Test texts
        y_test: True labels
        y_pred: Predicted labels
        sentiment_map: Dictionary mapping labels to names
        max_samples: Number of samples to return
    
    Returns:
        DataFrame with misclassified samples
    """
    # Find misclassified indices
    misclassified_idx = np.where(y_test != y_pred)[0]
    
    if len(misclassified_idx) == 0:
        return pd.DataFrame()
    
    # Limit to max_samples
    sample_idx = misclassified_idx[:max_samples]
    
    # Create DataFrame
    sentiment_map_reverse = {v: k for k, v in sentiment_map.items()}
    
    misclassified = pd.DataFrame({
        'Text': X_test[sample_idx],
        'True Label': [sentiment_map_reverse.get(y, 'Unknown') for y in y_test[sample_idx]],
        'Predicted Label': [sentiment_map_reverse.get(y, 'Unknown') for y in y_pred[sample_idx]]
    })
    
    return misclassified

