from augmentation_methods import aitchison_mixup, compositional_cutmix, compositional_feature_dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import pandas as pd


def create_classifier(clf_name, params=None, force_cpu=True):
    """
    Create classifier instance with given parameters and CPU/GPU control.
    """
    if params is None:
        params = {}

    if clf_name == 'rf':
        if force_cpu:
            print("Using CPU RandomForest (sklearn)")
            return RandomForestClassifier(random_state=42, n_jobs=-1, **params)

        try:
            from cuml.ensemble import RandomForestClassifier as cuRF
            print("Using GPU RandomForest (cuML)")
            cuml_params = params.copy()
            cuml_params.setdefault('random_state', 42)
            return cuRF(**cuml_params)

        except ImportError:
            print("cuML not available, falling back to CPU RandomForest")
            return RandomForestClassifier(random_state=42, n_jobs=-1, **params)

    elif clf_name == 'xgb':
        from xgboost import XGBClassifier
        print("Using XGBoost Classifier")
        return XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **params)

    elif clf_name == 'lr':
        print("Using Logistic Regression")
        lr_params = params.copy()
        lr_params.setdefault('random_state', 42)
        lr_params.setdefault('max_iter', 1000)
        lr_params.setdefault('n_jobs', -1)
        return LogisticRegression(**lr_params)

    elif clf_name == 'svm':
        print("Using Support Vector Machine")
        svm_params = params.copy()
        svm_params.setdefault('random_state', 42)
        svm_params.setdefault('probability', True)  # Required for predict_proba
        return SVC(**svm_params)

    elif clf_name == 'nb':
        from sklearn.naive_bayes import GaussianNB
        print("Using Gaussian Naive Bayes")
        return GaussianNB(**params)

    else:
        raise ValueError(f"Unsupported classifier type: {clf_name}")


def get_classifier_supports_feature_analysis(clf_name):
    """
    Check if classifier supports SHAP and feature importance analysis
    """
    # Only RF and XGBoost support comprehensive feature analysis
    return clf_name in ['rf', 'xgb']


def apply_augmentation(X, y, method='none', aug_ratio=3.0):
    """
    Apply data augmentation methods
    """
    if method == 'none':
        return X.copy(), y.copy()
    
    n_aug_samples = int(len(X) * aug_ratio)
    
    if method == 'aitchison_mixup':
        X_aug, y_aug = aitchison_mixup(X, y, n_samples=n_aug_samples)
    elif method == 'feature_dropout':
        X_aug, y_aug = compositional_feature_dropout(X, y, n_samples=n_aug_samples)
    elif method == "cutmix":
        X_aug, y_aug = compositional_cutmix(X, y, n_samples=n_aug_samples)
    else:
        raise ValueError(f"Unknown augmentation method: {method}")
    
    # Combine original and augmented data
    X_combined = pd.concat([X, X_aug], ignore_index=True)
    y_combined = np.concatenate([y, y_aug])
    
    return X_combined, y_combined