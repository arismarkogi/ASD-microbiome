import matplotlib.pyplot as plt
import shap
import numpy as np
import seaborn as sns
import pandas as pd

from find_best_augmentation_method import find_best_augmentation_method



def create_shap_plots(shap_results, aggregated_shap, max_display=20):
    """Create comprehensive SHAP visualization plots"""
    
    plt.style.use('default')
    
    # 1. Feature Importance Comparison Across Augmentation Methods
    fig, axes = plt.subplots(1, len(aggregated_shap), figsize=(15, 8))
    if len(aggregated_shap) == 1:
        axes = [axes]
    
    for idx, (aug_method, agg_data) in enumerate(aggregated_shap.items()):
        if agg_data is not None:
            top_features = agg_data['feature_importance_df'].head(max_display)
            
            axes[idx].barh(range(len(top_features)), top_features['importance'],
                          xerr=top_features['importance_std'], capsize=3)
            axes[idx].set_yticks(range(len(top_features)))
            axes[idx].set_yticklabels(top_features['feature'])
            axes[idx].set_xlabel('Mean |SHAP Value|')
            axes[idx].set_title(f'{aug_method}\n({agg_data["n_folds"]} folds)')
            axes[idx].invert_yaxis()
        else:
            axes[idx].text(0.5, 0.5, f'No SHAP results\nfor {aug_method}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{aug_method}\n(No results)')
    
    plt.tight_layout()
    plt.suptitle('Feature Importance Across Augmentation Methods', y=1.02)
    plt.show()
    
    # 2. Individual fold SHAP plots for best performing method
    best_method = find_best_augmentation_method(shap_results)
    if best_method:
        print(f"\nCreating detailed SHAP plots for best method: {best_method}")
        
        for fold_idx, fold_data in shap_results[best_method].items():
            if 'shap_values' in fold_data and 'error' not in fold_data:
                print(f"\nFold {fold_idx + 1} SHAP Analysis:")
                print(f"Accuracy: {fold_data['accuracy']:.4f}")
                
                # Summary plot for this fold
                shap_values = fold_data['shap_values']
                X_test = fold_data['X_test']
                feature_names = fold_data['feature_names']
                
                try:
                    # Handle different SHAP value shapes
                    if len(shap_values.shape) == 3:
                        # Shape: (samples, features, classes)
                        # For SHAP plots, we typically want to use one class
                        if shap_values.shape[2] == 2:
                            # Binary classification - use positive class
                            plot_shap_values = shap_values[:, :, 1]
                        else:
                            # Multi-class - use first class or average
                            plot_shap_values = shap_values[:, :, 0]
                    else:
                        # Already 2D: (samples, features)
                        plot_shap_values = shap_values
                    
                    # Create a DataFrame for better visualization
                    if len(plot_shap_values.shape) == 2:
                        X_test_df = pd.DataFrame(X_test, columns=feature_names)
                        
                        # SHAP summary plot
                        plt.figure(figsize=(12, 8))
                        shap.summary_plot(plot_shap_values, X_test_df,
                                        max_display=max_display, show=False)
                        plt.title(f'SHAP Summary Plot - {best_method} - Fold {fold_idx + 1}')
                        plt.tight_layout()
                        plt.show()
                        
                        # SHAP bar plot
                        plt.figure(figsize=(12, 6))
                        shap.summary_plot(plot_shap_values, X_test_df, plot_type="bar",
                                        max_display=max_display, show=False)
                        plt.title(f'SHAP Feature Importance - {best_method} - Fold {fold_idx + 1}')
                        plt.tight_layout()
                        plt.show()
                        
                except Exception as e:
                    print(f"Could not create SHAP plots for fold {fold_idx + 1}: {str(e)}")
                    print(f"SHAP values shape: {shap_values.shape}")
    
    # 3. Cross-fold consistency analysis
    create_shap_consistency_plot(shap_results, max_display=max_display)

def create_shap_consistency_plot(shap_results, max_display=20):
    """Create a plot showing feature importance consistency across folds"""
    
    for aug_method, fold_results in shap_results.items():
        # Extract feature importance for each fold
        fold_importances = []
        feature_names = None
        
        for fold_idx, fold_data in fold_results.items():
            if 'shap_values' in fold_data and 'error' not in fold_data:
                shap_vals = fold_data['shap_values']
                
                # Handle different SHAP value shapes - FIXED VERSION
                if len(shap_vals.shape) == 3:
                    # Shape: (samples, features, classes)
                    # First average across samples, then handle classes
                    shap_per_sample = np.mean(np.abs(shap_vals), axis=0)  # Shape: (features, classes)
                    
                    # For binary classification, typically use class 1, or take mean across classes
                    if shap_per_sample.shape[1] == 2:
                        importance = shap_per_sample[:, 1]  # Use positive class
                    else:
                        importance = np.mean(shap_per_sample, axis=1)  # Average across classes
                        
                elif len(shap_vals.shape) == 2:
                    # Shape: (samples, features)
                    importance = np.mean(np.abs(shap_vals), axis=0)
                else:
                    # Shape: (features,)
                    importance = np.abs(shap_vals)
                    
                fold_importances.append(importance)
                if feature_names is None:
                    feature_names = fold_data['feature_names']
        
        if len(fold_importances) < 2:
            print(f"Need at least 2 folds for consistency analysis of {aug_method}")
            continue
        
        # Check for shape consistency before creating DataFrame
        shapes = [imp.shape for imp in fold_importances]
        if len(set(shapes)) > 1:
            print(f"Warning: Inconsistent shapes in {aug_method}: {set(shapes)}")
            print("Skipping consistency plot for this method")
            continue
        
        # Create DataFrame
        importance_df = pd.DataFrame(fold_importances,
                                   columns=feature_names,
                                   index=[f'Fold {i+1}' for i in range(len(fold_importances))])
        
        # Get top features based on mean importance
        mean_importance = importance_df.mean(axis=0)
        top_features = mean_importance.nlargest(max_display).index
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(importance_df[top_features].T,
                    annot=True, fmt='.3f', cmap='viridis',
                    cbar_kws={'label': 'Mean |SHAP Value|'})
        plt.title(f'Feature Importance Consistency - {aug_method}')
        plt.xlabel('Fold')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

def calculate_shap_importance(shap_vals):
    """
    Helper function to calculate feature importance from SHAP values
    Handles different SHAP value shapes consistently
    """
    if len(shap_vals.shape) == 3:
        # Shape: (samples, features, classes)
        # First average across samples, then handle classes
        shap_per_sample = np.mean(np.abs(shap_vals), axis=0)  # Shape: (features, classes)
        
        # For binary classification, typically use class 1, or take mean across classes
        if shap_per_sample.shape[1] == 2:
            importance = shap_per_sample[:, 1]  # Use positive class
        else:
            importance = np.mean(shap_per_sample, axis=1)  # Average across classes
            
    elif len(shap_vals.shape) == 2:
        # Shape: (samples, features)
        importance = np.mean(np.abs(shap_vals), axis=0)
    else:
        # Shape: (features,)
        importance = np.abs(shap_vals)
    
    return importance