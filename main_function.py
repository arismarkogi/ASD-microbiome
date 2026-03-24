import os
import numpy as np
from run_integrated_rf_shap_analysis import  run_integrated_classifier_analysis
from nested_cv_util import get_classifier_supports_feature_analysis


def main_function(merged_df, X_raw_significant, y, base_dir="classifier_experiments",
                 classifier_name='rf',
                 augmentation_methods=['none', 'aitchison_mixup', 'feature_dropout', 'cutmix'],
                 outer_cv=5, inner_cv=3, random_state=42, force_cpu=True,
                 max_display=15, save_results=True):
    """
    Main function for classifier analysis on full dataset.
    
    Args:
        merged_df: DataFrame containing the cohort information
        X_raw_significant: Feature matrix
        y: Target labels
        base_dir: Base directory for saving results
        classifier_name: Name of classifier to use ('rf', 'xgb', 'lr', 'svm')
        augmentation_methods: List of augmentation methods to use
        outer_cv: Number of outer CV folds
        inner_cv: Number of inner CV folds
        random_state: Random seed
        force_cpu: Whether to force CPU usage
        max_display: Maximum features to display
        save_results: Whether to save results
        
    Returns:
        Results from the full dataset analysis
    """
    # Create base results directory
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Running Nested CV with {classifier_name.upper()} on the FULL dataset")
    print(f"{'='*50}")
    print(f"Dataset shape: {X_raw_significant.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Classifier: {classifier_name.upper()}")
    print(f"Augmentation methods: {augmentation_methods}")
    
    # Create directory for full dataset analysis
    full_dataset_dir = os.path.join(base_dir, f"{classifier_name}_all")
    os.makedirs(full_dataset_dir, exist_ok=True)
    
    # Run the integrated analysis
    results_full = run_integrated_classifier_analysis(
        X_raw_significant, y,
        classifier_name=classifier_name,
        augmentation_methods=augmentation_methods,
        outer_cv=outer_cv,
        inner_cv=inner_cv,
        random_state=random_state,
        force_cpu=force_cpu,
        max_display=max_display,
        save_results=save_results,
        experiment_name=f"{classifier_name}_all",
        results_dir=full_dataset_dir
    )
    
    print(f"\nCompleted analysis for FULL dataset using {classifier_name.upper()}.")
    
    # Print final summary of what was generated
    supports_feature_analysis = get_classifier_supports_feature_analysis(classifier_name)
    print(f"\nGenerated outputs:")
    print(f"- Performance metrics and confusion matrices: Yes")
    print(f"- SHAP analysis plots: {'Yes' if supports_feature_analysis else 'No'}")
    print(f"- Feature importance plots: {'Yes' if classifier_name in ['rf', 'xgb'] else 'No'}")
    print(f"- Results saved in: {full_dataset_dir}")
    
    return results_full


# Alternative function for running multiple classifiers
def run_multiple_classifiers(merged_df, X_raw_significant, y, 
                           base_dir="multi_classifier_experiments",
                           classifiers=['rf', 'xgb', 'lr', 'svm'],
                           augmentation_methods=['none', 'aitchison_mixup', 'feature_dropout', 'cutmix'],
                           outer_cv=5, inner_cv=3, random_state=42, force_cpu=True,
                           max_display=15, save_results=True):
    """
    Run analysis with multiple classifiers for comparison
    
    Args:
        Same as main_function, but classifiers is a list of classifier names
        
    Returns:
        dict: Results for each classifier
    """
    all_results = {}
    
    print(f"\n{'='*60}")
    print(f"RUNNING MULTI-CLASSIFIER ANALYSIS")
    print(f"{'='*60}")
    print(f"Classifiers: {', '.join([c.upper() for c in classifiers])}")
    
    for classifier_name in classifiers:
        print(f"\n{'='*50}")
        print(f"STARTING ANALYSIS FOR {classifier_name.upper()}")
        print(f"{'='*50}")
        
        try:
            results = main_function(
                merged_df, X_raw_significant, y,
                base_dir=base_dir,
                classifier_name=classifier_name,
                augmentation_methods=augmentation_methods,
                outer_cv=outer_cv,
                inner_cv=inner_cv,
                random_state=random_state,
                force_cpu=force_cpu,
                max_display=max_display,
                save_results=save_results
            )
            all_results[classifier_name] = results
            print(f"✓ Successfully completed {classifier_name.upper()} analysis")
            
        except Exception as e:
            print(f"✗ Failed to complete {classifier_name.upper()} analysis: {str(e)}")
            all_results[classifier_name] = None
    
    # Print final comparison summary
    print(f"\n{'='*60}")
    print("MULTI-CLASSIFIER COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n{'Classifier':<12} {'Best Method':<20} {'Accuracy':<12} {'AUC':<12} {'F1':<12}")
    print("-" * 68)
    
    for classifier_name, results in all_results.items():
        if results is not None:
            cv_results = results[0]  # cv_results is always first in tuple
            
            # Find best performing method
            best_method = None
            best_acc = -1
            for method, metrics in cv_results.items():
                acc_mean = np.mean(metrics['accuracy'])
                if acc_mean > best_acc:
                    best_acc = acc_mean
                    best_method = method
            
            if best_method:
                best_metrics = cv_results[best_method]
                acc_mean = np.mean(best_metrics['accuracy'])
                auc_mean = np.mean(best_metrics['auc'])
                f1_mean = np.mean(best_metrics['f1'])
                
                print(f"{classifier_name.upper():<12} {best_method:<20} {acc_mean:.4f}      {auc_mean:.4f}     {f1_mean:.4f}")
        else:
            print(f"{classifier_name.upper():<12} {'FAILED':<20} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    return all_results