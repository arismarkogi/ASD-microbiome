

def get_hyperparameter_grid(clf_name):
    """
    Get hyperparameter grid for different classifiers
    """
    if clf_name == 'rf':
        return get_rf_hyperparameter_grid()
    elif clf_name == 'xgb':
        return get_xgb_hyperparameter_grid()
    elif clf_name == 'lr':
        return get_lr_hyperparameter_grid()
    elif clf_name == 'svm':
        return get_svm_hyperparameter_grid()
    else:
        raise ValueError(f"No hyperparameter grid defined for: {clf_name}")


def get_rf_hyperparameter_grid():
    """
    Focused hyperparameter grid for Random Forest (~24 combinations)
    """
    return {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [ 10, 20, 30],
        'min_samples_split': [2, 5],
    }


def get_xgb_hyperparameter_grid():
    """XGBoost hyperparameter grid"""
    return {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.05, 0.1, 0.2],
    }


def get_lr_hyperparameter_grid():
    """Logistic Regression hyperparameter grid (reduced)"""
    return {
        'C': [0.1, 1],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }


def get_svm_hyperparameter_grid():
    """SVM hyperparameter grid (reduced)"""
    return {
        'C': [0.1, 1],
        'kernel': ['rbf'],
        'gamma': ['scale', 0.1]
    }