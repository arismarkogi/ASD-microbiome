import pandas as pd
import numpy as np
from datetime import datetime

def save_all_results(cv_results, shap_results, aggregated_shap, best_params_storage, experiment_name):
    """Save all results to CSV files"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{experiment_name}_{timestamp}"
    
    # 1. Save CV results summary (unchanged)
    cv_summary = {}
    for method, metrics in cv_results.items():
        cv_summary[method] = {
            'accuracy_mean': np.mean(metrics['accuracy']),
            'accuracy_std': np.std(metrics['accuracy']),
            'auc_mean': np.mean(metrics['auc']),
            'auc_std': np.std(metrics['auc']),
            'f1_mean': np.mean(metrics['f1']),
            'f1_std': np.std(metrics['f1'])
        }
    
    pd.DataFrame(cv_summary).T.to_csv(f"{base_filename}_cv_summary.csv")
    
    # 2. Save detailed CV results (all folds)
    cv_detailed = []
    for method, metrics in cv_results.items():
        for fold_idx, (acc, auc, f1) in enumerate(zip(metrics['accuracy'], metrics['auc'], metrics['f1'])):
            cv_detailed.append({
                'method': method,
                'fold': fold_idx,
                'accuracy': acc,
                'auc': auc,
                'f1': f1
            })
    
    pd.DataFrame(cv_detailed).to_csv(f"{base_filename}_cv_detailed.csv", index=False)
    
    # 3. Save aggregated SHAP results (unchanged)
    for method, agg_data in aggregated_shap.items():
        if agg_data is not None:
            agg_data['feature_importance_df'].to_csv(
                f"{base_filename}_shap_importance_{method}.csv", index=False
            )
    
    # 4. Save detailed SHAP results (fold-level)
    shap_detailed = []
    for method, method_data in shap_results.items():
        for fold_key, fold_data in method_data.items():
            if 'feature_importance' in fold_data:
                for feature, importance in fold_data['feature_importance'].items():
                    shap_detailed.append({
                        'method': method,
                        'fold': fold_key,
                        'feature': feature,
                        'importance': importance
                    })
    
    if shap_detailed:
        pd.DataFrame(shap_detailed).to_csv(f"{base_filename}_shap_detailed.csv", index=False)
    
    # 5. Save best parameters
    best_params_list = []
    for method, params in best_params_storage.items():
        for param_name, param_value in params.items():
            best_params_list.append({
                'method': method,
                'parameter': param_name,
                'value': str(param_value)  # Convert to string to handle various types
            })
    
    if best_params_list:
        pd.DataFrame(best_params_list).to_csv(f"{base_filename}_best_params.csv", index=False)
    
    # 6. Save experiment metadata
    experiment_info = pd.DataFrame([{
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'base_filename': base_filename,
        'methods_tested': ', '.join(cv_results.keys()),
        'total_methods': len(cv_results),
        'cv_folds': len(next(iter(cv_results.values()))['accuracy']) if cv_results else 0
    }])
    
    experiment_info.to_csv(f"{base_filename}_experiment_info.csv", index=False)
    
    print(f"Results saved as CSV files with base filename: {base_filename}")
    print("Files created:")
    print(f"  - {base_filename}_cv_summary.csv")
    print(f"  - {base_filename}_cv_detailed.csv")
    print(f"  - {base_filename}_shap_importance_[method].csv")
    print(f"  - {base_filename}_shap_detailed.csv")
    print(f"  - {base_filename}_best_params.csv")
    print(f"  - {base_filename}_experiment_info.csv")
    
    return base_filename