import os
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

from save_all_results import save_all_results
from create_and_save_all_plots import create_and_save_all_plots
from nested_cv_with_shap_analysis import nested_cv_with_shap_analysis, nested_cv_with_conditional_analysis
from nested_cv_util import get_classifier_supports_feature_analysis


#def run_integrated_rf_shap_analysis(X, y, 
#                                    augmentation_methods=['none', 'aitchison_mixup', 'feature_dropout', 'cutmix'],
#                                    outer_cv=5, inner_cv=3, random_state=42,
#                                    force_cpu=False, max_display=20, 
#                                    save_results=True, experiment_name="rf_shap_analysis", results_dir=None):
#     """
#     Main function to run integrated Random Forest CV with SHAP analysis
    
#     Args:
#         X: Feature matrix
#         y: Target labels
#         augmentation_methods: List of augmentation methods to use
#         outer_cv: Number of outer CV folds
#         inner_cv: Number of inner CV folds
#         random_state: Random seed
#         force_cpu: Whether to force CPU usage
#         max_display: Maximum features to display
#         save_results: Whether to save results
#         experiment_name: Name for the experiment
#         results_dir: Directory to save results (if None, uses current directory)
    
#     Returns:
#         tuple: (cv_results, shap_results, aggregated_shap, best_params_storage)
#     """
    
#     # Change to results directory if specified
#     original_dir = None
#     if results_dir:
#         original_dir = os.getcwd()
#         os.chdir(results_dir)
    
#     try:
#         print(f"{'='*80}")
#         print("INTEGRATED RANDOM FOREST CV WITH SHAP ANALYSIS")
#         print(f"{'='*80}")
        
#         # Convert to relative abundance if needed
#         if not np.allclose(X.sum(axis=1), 1.0):
#             print("Converting to relative abundance...")
#             X = X.div(X.sum(axis=1), axis=0)
#             print("Data converted to relative abundance (compositional data)")
        
#         # Handle missing values if any
#         if X.isnull().any().any():
#             print("Handling missing values...")
#             imputer = SimpleImputer(strategy='median')
#             X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
#             X = X_imputed
#             print("Missing values imputed with median")
        
#         # Ensure all values are positive (required for compositional data)
#         X = X.clip(lower=1e-10)
        
#         # Run the nested CV with SHAP analysis
#         cv_results, shap_results, aggregated_shap, best_params_storage, splitting_features_storage, aggregated_splitting, confusion_matrices = nested_cv_with_shap_analysis(
#             X, y,
#             augmentation_methods=augmentation_methods,
#             outer_cv=outer_cv,
#             inner_cv=inner_cv,
#             random_state=random_state,
#             force_cpu=force_cpu,
#             max_display=max_display
#         )
        
#         # Save results if requested
#         if save_results:
#             base_filename = save_all_results(
#                 cv_results, shap_results, aggregated_shap, 
#                 best_params_storage, experiment_name
#             )
#             print(f"\nResults saved with base filename: {base_filename}")
        
#         # Create and save plots
#         if save_results:
#             create_and_save_all_plots(cv_results, shap_results, aggregated_shap, 
#                                      aggregated_splitting, max_display=max_display)
        
#         # Print final summary
#         print(f"\n{'='*80}")
#         print("FINAL SUMMARY")
#         print(f"{'='*80}")
        
#         print("\nPerformance Summary:")
#         for method, metrics in cv_results.items():
#             acc_mean = np.mean(metrics['accuracy'])
#             acc_std = np.std(metrics['accuracy'])
#             print(f"{method}: {acc_mean:.4f} ± {acc_std:.4f} accuracy")
        
#         print("\nTop Features by Augmentation Method:")
#         for method, agg_data in aggregated_shap.items():
#             if agg_data is not None:
#                 top_5_features = agg_data['feature_importance_df'].head(5)['feature'].tolist()
#                 print(f"{method}: {', '.join(top_5_features)}")
        
#         return cv_results, shap_results, aggregated_shap, best_params_storage
    
#     finally:
#         # Always change back to original directory
#         if original_dir:
#             os.chdir(original_dir)

def run_integrated_classifier_analysis(X, y, classifier_name='rf',
                                      augmentation_methods=['none', 'aitchison_mixup', 'feature_dropout', 'cutmix'],
                                      outer_cv=5, inner_cv=3, random_state=42,
                                      force_cpu=False, max_display=20, 
                                      save_results=True, experiment_name=None, results_dir=None):
    """
    Main function to run integrated classifier CV with conditional feature analysis
    
    Args:
        X: Feature matrix
        y: Target labels
        classifier_name: Name of classifier ('rf', 'xgb', 'lr', 'svm')
        augmentation_methods: List of augmentation methods to use
        outer_cv: Number of outer CV folds
        inner_cv: Number of inner CV folds
        random_state: Random seed
        force_cpu: Whether to force CPU usage
        max_display: Maximum features to display
        save_results: Whether to save results
        experiment_name: Name for the experiment (if None, uses classifier_name)
        results_dir: Directory to save results (if None, uses current directory)
    
    Returns:
        tuple: Results vary based on classifier support for feature analysis
               - For RF/XGBoost: (cv_results, shap_results, aggregated_shap, best_params_storage, 
                                  splitting_features_storage, aggregated_splitting, confusion_matrices)
               - For LR/SVM: (cv_results, None, None, best_params_storage, None, None, confusion_matrices)
    """
    
    # Set default experiment name
    if experiment_name is None:
        experiment_name = f"{classifier_name}_analysis"
    
    # Change to results directory if specified
    original_dir = None
    if results_dir:
        original_dir = os.getcwd()
        os.chdir(results_dir)
    
    try:
        print(f"{'='*80}")
        print(f"INTEGRATED {classifier_name.upper()} CV WITH CONDITIONAL FEATURE ANALYSIS")
        print(f"{'='*80}")
        
        # Check classifier capabilities
        supports_feature_analysis = get_classifier_supports_feature_analysis(classifier_name)
        print(f"Classifier: {classifier_name.upper()}")
        print(f"Feature analysis support: {'Yes' if supports_feature_analysis else 'No'}")
        
        # Convert to relative abundance if needed (for compositional data)
        if  not np.allclose(X.sum(axis=1), 1.0):
            print("Converting to relative abundance...")
            X = X.div(X.sum(axis=1), axis=0)
            print("Data converted to relative abundance (compositional data)")
        
        # Handle missing values if any
        if X.isnull().any().any():
            print("Handling missing values...")
            imputer = SimpleImputer(strategy='median')
            X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
            X = X_imputed
            print("Missing values imputed with median")
        
        # Ensure all values are positive for compositional data (tree-based models)
        if classifier_name in ['rf', 'xgb']:
            X = X.clip(lower=1e-10)
        
        # Run the nested CV with conditional analysis
        results = nested_cv_with_conditional_analysis(
            X, y,
            classifier_name=classifier_name,
            augmentation_methods=augmentation_methods,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            random_state=random_state,
            force_cpu=force_cpu,
            max_display=max_display
        )
        
        # Unpack results based on what's returned
        if supports_feature_analysis:
            (cv_results, shap_results, aggregated_shap, best_params_storage, 
             splitting_features_storage, aggregated_splitting, confusion_matrices) = results
        else:
            (cv_results, shap_results, aggregated_shap, best_params_storage, 
             splitting_features_storage, aggregated_splitting, confusion_matrices) = results
            # Note: shap_results, aggregated_shap, splitting_features_storage, aggregated_splitting will be None
        
        # Save results if requested
        if save_results:
            base_filename = save_all_results(
                cv_results, shap_results, aggregated_shap, 
                best_params_storage, experiment_name
            )
            print(f"\nResults saved with base filename: {base_filename}")
        
        # Create and save plots
        if save_results:
            create_and_save_all_plots(cv_results, shap_results, aggregated_shap, 
                                    aggregated_splitting, classifier_name, max_display=max_display)
        
        # Print final summary
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY - {classifier_name.upper()}")
        print(f"{'='*80}")
        
        print("\nPerformance Summary:")
        for method, metrics in cv_results.items():
            acc_mean = np.mean(metrics['accuracy'])
            acc_std = np.std(metrics['accuracy'])
            auc_mean = np.mean(metrics['auc'])
            auc_std = np.std(metrics['auc'])
            f1_mean = np.mean(metrics['f1'])
            f1_std = np.std(metrics['f1'])
            print(f"{method}: Acc={acc_mean:.4f}±{acc_std:.4f}, AUC={auc_mean:.4f}±{auc_std:.4f}, F1={f1_mean:.4f}±{f1_std:.4f}")
        
        # Feature analysis summary (only if supported)
        if supports_feature_analysis and aggregated_shap:
            print("\nTop Features by Augmentation Method (SHAP):")
            for method, agg_data in aggregated_shap.items():
                if agg_data is not None:
                    top_5_features = agg_data['feature_importance_df'].head(5)['feature'].tolist()
                    print(f"{method}: {', '.join(top_5_features)}")
        elif not supports_feature_analysis:
            print(f"\nFeature analysis not available for {classifier_name.upper()}")
            print("Only performance metrics and confusion matrices are generated.")
        
        return results
    
    finally:
        # Always change back to original directory
        if original_dir:
            os.chdir(original_dir)
