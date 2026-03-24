import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def create_aggregated_confusion_matrix(cv_results, confusion_matrices, save_dir="confusion_matrices"):
    """Create and save aggregated confusion matrices across all folds"""
    os.makedirs(save_dir, exist_ok=True)
    
    for aug_method in cv_results.keys():
        if aug_method in confusion_matrices:
            total_cm = sum(confusion_matrices[aug_method])
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', 
                        square=True, cbar_kws={'shrink': 0.8})
            
            plt.title(f'Aggregated Confusion Matrix - {aug_method} (All Folds)')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            filename = f"{save_dir}/confusion_matrix_{aug_method}_aggregated.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

def plot_and_save_confusion_matrix(y_true, y_pred, aug_method, fold_idx, save_dir="confusion_matrices"):
    """Plot and save confusion matrix for a specific fold and augmentation method"""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                square=True, cbar_kws={'shrink': 0.8})
    
    plt.title(f'Confusion Matrix - {aug_method} (Fold {fold_idx + 1})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    filename = f"{save_dir}/confusion_matrix_{aug_method}_fold_{fold_idx + 1}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm, filename