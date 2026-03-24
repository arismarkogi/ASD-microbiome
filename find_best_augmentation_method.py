import numpy as np

def find_best_augmentation_method(shap_results):
    """Find the best performing augmentation method based on average accuracy"""
    method_performance = {}
    
    for aug_method, fold_results in shap_results.items():
        accuracies = []
        for fold_idx, fold_data in fold_results.items():
            if 'accuracy' in fold_data:
                accuracies.append(fold_data['accuracy'])
        
        if accuracies:
            method_performance[aug_method] = np.mean(accuracies)
    
    if method_performance:
        return max(method_performance, key=method_performance.get)
    return None