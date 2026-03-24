import shap
from collections import defaultdict
import numpy as np
import pandas as pd


def get_shap_values(model, X_train, X_test, model_type, force_cpu=False):
    """Get SHAP values for different model types"""

    # Convert cuML models to sklearn for SHAP compatibility
    if 'cuml' in str(type(model)) and not force_cpu:
        print("Converting cuML model to sklearn for SHAP analysis")
        # For cuML models, we need to create an sklearn equivalent
        # This is a limitation - cuML models don't work directly with SHAP
        from sklearn.ensemble import RandomForestClassifier
        
        # Get predictions from cuML model to train sklearn equivalent
        train_predictions = model.predict(X_train)
        
        sklearn_model = RandomForestClassifier(
            n_estimators=getattr(model, 'n_estimators', 100),
            max_depth=getattr(model, 'max_depth', None),
            min_samples_split=getattr(model, 'min_samples_split', 2),
            random_state=42,
            n_jobs=-1
        )
        sklearn_model.fit(X_train, train_predictions)
        model = sklearn_model

    # Choose appropriate SHAP explainer based on model type
    if model_type == 'rf':
        # TreeExplainer for Random Forest
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # For binary classification, TreeExplainer returns values for both classes
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # Use positive class

    elif model_type == 'xgb':
        # TreeExplainer for XGBoost
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

    elif model_type == 'nb':
        # For Naive Bayes, use LinearExplainer or Permutation
        try:
            # Try LinearExplainer first (faster)
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test)
        except:
            # Fall back to Permutation explainer
            explainer = shap.PermutationExplainer(model.predict_proba, X_train)
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class

    else:
        # Default to Permutation explainer for unknown models
        explainer = shap.PermutationExplainer(model.predict_proba, X_train)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class

    return shap_values, explainer

def aggregate_shap_results(shap_results, feature_names):
    """Aggregate SHAP values across all folds and methods with robust error handling"""
    aggregated = defaultdict(dict)
    
    for aug_method, fold_results in shap_results.items():
        print(f"\nAggregating SHAP results for {aug_method}:")
        
        # Collect all SHAP values for this augmentation method
        method_shap_values = []
        method_feature_importance = []
        successful_folds = 0
        expected_shape = None
        
        for fold_idx, fold_data in fold_results.items():
            if 'shap_values' in fold_data and 'error' not in fold_data:
                try:
                    shap_vals = fold_data['shap_values']
                    
                    # Debug: print shape information
                    print(f"  Fold {fold_idx}: SHAP values shape = {shap_vals.shape}, type = {type(shap_vals)}")
                    
                    # Calculate mean absolute SHAP values for feature importance
                    if len(shap_vals.shape) == 3:
                        # Shape: (samples, features, classes)
                        # First average across samples, then handle classes
                        shap_per_sample = np.mean(np.abs(shap_vals), axis=0)  # Shape: (features, classes)
                        
                        # For binary classification, typically use class 1, or take mean across classes
                        if shap_per_sample.shape[1] == 2:
                            fold_importance = shap_per_sample[:, 1]  # Use positive class
                        else:
                            fold_importance = np.mean(shap_per_sample, axis=1)  # Average across classes
                            
                    elif len(shap_vals.shape) == 2:
                        # Shape: (samples, features)
                        fold_importance = np.mean(np.abs(shap_vals), axis=0)
                    else:
                        # Shape: (features,)
                        fold_importance = np.abs(shap_vals)
                    
                    # Check shape consistency
                    if expected_shape is None:
                        expected_shape = fold_importance.shape
                        print(f"  Expected feature importance shape: {expected_shape}")
                    elif fold_importance.shape != expected_shape:
                        print(f"  WARNING: Shape mismatch in fold {fold_idx}!")
                        print(f"  Expected: {expected_shape}, Got: {fold_importance.shape}")
                        
                        # Try to handle the mismatch
                        if fold_importance.shape[0] != expected_shape[0]:
                            print(f"  Skipping fold {fold_idx} due to feature count mismatch")
                            continue
                        else:
                            # Try to reshape if possible
                            try:
                                fold_importance = fold_importance.reshape(expected_shape)
                                print(f"  Successfully reshaped to {expected_shape}")
                            except:
                                print(f"  Could not reshape, skipping fold {fold_idx}")
                                continue
                    
                    # Verify feature count matches
                    if len(feature_names) != fold_importance.shape[0]:
                        print(f"  WARNING: Feature count mismatch!")
                        print(f"  Expected {len(feature_names)} features, got {fold_importance.shape[0]}")
                        
                        if fold_importance.shape[0] > len(feature_names):
                            # Truncate to match feature names
                            fold_importance = fold_importance[:len(feature_names)]
                            print(f"  Truncated to {len(feature_names)} features")
                        else:
                            print(f"  Skipping fold {fold_idx} due to insufficient features")
                            continue
                    
                    method_shap_values.append(shap_vals)
                    method_feature_importance.append(fold_importance)
                    successful_folds += 1
                    print(f"  Successfully processed fold {fold_idx}")
                    
                except Exception as e:
                    print(f"  ERROR processing fold {fold_idx}: {str(e)}")
                    print(f"  SHAP values info: shape={getattr(shap_vals, 'shape', 'N/A')}, dtype={getattr(shap_vals, 'dtype', 'N/A')}")
                    continue
            else:
                print(f"  Skipping fold {fold_idx}: no valid SHAP values")
        
        if method_feature_importance:
            try:
                # Print shapes before aggregation for debugging
                print(f"  Feature importance shapes: {[arr.shape for arr in method_feature_importance]}")
                
                # Check if all arrays have the same shape
                shapes = [arr.shape for arr in method_feature_importance]
                if len(set(shapes)) > 1:
                    print(f"  ERROR: Inconsistent shapes detected: {set(shapes)}")
                    print(f"  Attempting to fix...")
                    
                    # Find the most common shape
                    from collections import Counter
                    shape_counts = Counter(shapes)
                    target_shape = shape_counts.most_common(1)[0][0]
                    print(f"  Target shape: {target_shape}")
                    
                    # Filter arrays to only include those with the target shape
                    filtered_importance = []
                    for i, arr in enumerate(method_feature_importance):
                        if arr.shape == target_shape:
                            filtered_importance.append(arr)
                        else:
                            print(f"  Excluded array {i} with shape {arr.shape}")
                    
                    method_feature_importance = filtered_importance
                    successful_folds = len(method_feature_importance)
                    print(f"  Using {successful_folds} folds after filtering")
                
                if method_feature_importance:
                    # Convert to numpy array for aggregation
                    importance_array = np.array(method_feature_importance)
                    print(f"  Final importance array shape: {importance_array.shape}")
                    
                    # Aggregate across folds
                    aggregated[aug_method] = {
                        'mean_abs_shap': np.mean(importance_array, axis=0),
                        'std_abs_shap': np.std(importance_array, axis=0),
                        'feature_names': feature_names,
                        'n_folds': successful_folds
                    }
                    
                    # Create feature importance ranking
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': aggregated[aug_method]['mean_abs_shap'],
                        'importance_std': aggregated[aug_method]['std_abs_shap']
                    }).sort_values('importance', ascending=False)
                    
                    aggregated[aug_method]['feature_importance_df'] = importance_df
                    print(f"  Successfully aggregated {successful_folds} folds")
                    print(f"  Top 5 features: {importance_df.head()['feature'].tolist()}")
                else:
                    print(f"  No valid arrays remaining after filtering")
                    aggregated[aug_method] = None
                    
            except Exception as e:
                print(f"  ERROR during aggregation: {str(e)}")
                print(f"  Method feature importance info:")
                for i, arr in enumerate(method_feature_importance):
                    print(f"    Array {i}: shape={arr.shape}, dtype={arr.dtype}")
                aggregated[aug_method] = None
        else:
            print(f"  No successful SHAP analyses for {aug_method}")
            aggregated[aug_method] = None
    
    return dict(aggregated)


# Additional debugging function
def debug_shap_structure(shap_results):
    """Debug function to inspect the structure of SHAP results"""
    print("\n" + "="*60)
    print("DEBUGGING SHAP STRUCTURE")
    print("="*60)
    
    for aug_method, fold_results in shap_results.items():
        print(f"\nMethod: {aug_method}")
        print(f"Number of folds: {len(fold_results)}")
        
        for fold_idx, fold_data in fold_results.items():
            print(f"  Fold {fold_idx}:")
            if fold_data is None:
                print(f"    Data: None")
            elif isinstance(fold_data, dict):
                print(f"    Keys: {list(fold_data.keys())}")
                if 'shap_values' in fold_data:
                    shap_vals = fold_data['shap_values']
                    print(f"    SHAP values type: {type(shap_vals)}")
                    print(f"    SHAP values shape: {getattr(shap_vals, 'shape', 'N/A')}")
                    print(f"    SHAP values dtype: {getattr(shap_vals, 'dtype', 'N/A')}")
                if 'error' in fold_data:
                    print(f"    Error: {fold_data['error']}")
            else:
                print(f"    Data type: {type(fold_data)}")
    
    print("="*60)