import numpy as np
import shap
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from hyperparameter_grid import get_rf_hyperparameter_grid
from nested_cv_util import apply_augmentation, create_classifier
from shap_plots import create_shap_plots
from shap_util import get_shap_values, aggregate_shap_results
from confusion_matrix_util import create_aggregated_confusion_matrix, plot_and_save_confusion_matrix
from splitting_analysis import perform_enhanced_splitting_analysis, aggregate_splitting_features

def nested_cv_with_shap_analysis(X, y, 
                                augmentation_methods=['none', 'aitchison_mixup', 'feature_dropout', 'cutmix'],
                                outer_cv=5, inner_cv=3, random_state=42, 
                                force_cpu=False, max_display=20):
    """
    Perform nested cross-validation with Random Forest and comprehensive SHAP analysis
    """
    print(f"{'='*80}")
    print("NESTED CV WITH RANDOM FOREST AND SHAP ANALYSIS")
    print(f"{'='*80}")
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Augmentation methods: {augmentation_methods}")
    
    # Initialize results storage
    cv_results = defaultdict(lambda: defaultdict(list))
    shap_results = defaultdict(lambda: defaultdict(dict))
    best_params_storage = defaultdict(lambda: defaultdict(list))
    confusion_matrices = defaultdict(list)
    
    # Initialize SHAP explainer
    shap.initjs()
    
    # Outer CV loop
    outer_skf = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"OUTER FOLD {fold_idx + 1}/{outer_cv}")
        print(f"{'='*60}")
        
        # Split data for this outer fold
        X_train_outer = X.iloc[train_idx]
        X_test_outer = X.iloc[test_idx]
        y_train_outer = y[train_idx]
        y_test_outer = y[test_idx]
        
        print(f"Train size: {len(X_train_outer)}, Test size: {len(X_test_outer)}")
        
        # Test each augmentation method
        for aug_method in augmentation_methods:
            print(f"\n{'-'*40}")
            print(f"AUGMENTATION METHOD: {aug_method}")
            print(f"{'-'*40}")
            
            # Apply augmentation
            if aug_method == 'none':
                X_train_aug = X_train_outer.copy()
                y_train_aug = y_train_outer.copy()
            else:
                X_train_aug, y_train_aug = apply_augmentation(
                    X_train_outer, y_train_outer, method=aug_method, aug_ratio=3.0
                )
            
            print(f"Training samples after augmentation: {len(X_train_aug)}")
            print(f"Class distribution: {np.bincount(y_train_aug)}")
            
            # Inner CV for hyperparameter tuning
            inner_skf = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_state)
            
            # Create base classifier
            base_rf = create_classifier('rf', params={}, force_cpu=force_cpu)
            
            # Hyperparameter grid
            param_grid = get_rf_hyperparameter_grid()
            
            # Grid search
            print("Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                estimator=base_rf,
                param_grid=param_grid,
                cv=inner_skf,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            # Fit grid search
            grid_search.fit(X_train_aug.values, y_train_aug)
            best_params = grid_search.best_params_
            
            print(f"Best parameters: {best_params}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Store best parameters
            best_params_storage[aug_method][fold_idx] = best_params
            
            # Train final model with best parameters
            final_rf = create_classifier('rf', params=best_params, force_cpu=force_cpu)
            
            # Handle data preprocessing for different model types
            if 'cuml' in str(type(final_rf)) and not force_cpu:
                X_train_processed = X_train_aug.values.astype(np.float32)
                X_test_processed = X_test_outer.values.astype(np.float32)
            else:
                X_train_processed = X_train_aug.values
                X_test_processed = X_test_outer.values
            
            # Train final model
            final_rf.fit(X_train_processed, y_train_aug)
            
            # Evaluate on test set
            y_pred = final_rf.predict(X_test_processed)
            y_pred_proba = final_rf.predict_proba(X_test_processed)
            
            # Handle probability output for different model types
            if len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 2:
                y_pred_proba_pos = y_pred_proba[:, 1]
            else:
                y_pred_proba_pos = y_pred_proba
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_outer, y_pred)
            auc = roc_auc_score(y_test_outer, y_pred_proba_pos)
            f1 = f1_score(y_test_outer, y_pred, average='weighted')
            cm, cm_filename = plot_and_save_confusion_matrix(
                y_test_outer, y_pred, aug_method, fold_idx
            )
            confusion_matrices[aug_method].append(cm)
            
            # Store CV results
            cv_results[aug_method]['accuracy'].append(accuracy)
            cv_results[aug_method]['auc'].append(auc)
            cv_results[aug_method]['f1'].append(f1)
            
            print(f"\nPerformance on test set:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC: {auc:.4f}")
            print(f"  F1: {f1:.4f}")
            print(f"  Confusion Matrix:\n{cm}")
            print(f"  Confusion matrix saved: {cm_filename}")
            
            # SHAP Analysis
            print(f"\nPerforming SHAP analysis...")
            try:
                shap_values, explainer = get_shap_values(
                    final_rf, X_train_processed, X_test_processed, 'rf', force_cpu
                )
                
                # Store SHAP results
                shap_results[aug_method][fold_idx] = {
                    'shap_values': shap_values,
                    'explainer': explainer,
                    'X_test': X_test_processed,
                    'y_test': y_test_outer,
                    'y_pred': y_pred,
                    'feature_names': X.columns.tolist(),
                    'model_type': 'rf',
                    'aug_method': aug_method,
                    'accuracy': accuracy,
                    'auc': auc,
                    'f1': f1,
                    'best_params': best_params
                }
                
                print(f"SHAP analysis completed successfully")
                
            except Exception as e:
                print(f"SHAP analysis failed: {str(e)}")
                shap_results[aug_method][fold_idx] = {
                    'error': str(e),
                    'feature_names': X.columns.tolist(),
                    'model_type': 'rf',
                    'aug_method': aug_method,
                    'accuracy': accuracy,
                    'auc': auc,
                    'f1': f1,
                    'best_params': best_params
                }
            # SPLITTING FEATURES ANALYSIS
            print(f"\nPerforming splitting features analysis...")
            try:
                splitting_results = perform_enhanced_splitting_analysis(
                    final_rf, X_train_processed, y_train_aug, X.columns.tolist(), aug_method, fold_idx
                )
                
                # Store splitting results
                if fold_idx == 0:  # Initialize storage for first fold
                    splitting_features_storage = defaultdict(dict)
                
                splitting_features_storage[aug_method][fold_idx] = splitting_results
                
            except Exception as e:
                print(f"Splitting features analysis failed: {str(e)}")
                if fold_idx == 0:
                    splitting_features_storage = defaultdict(dict)
                splitting_features_storage[aug_method][fold_idx] = None
    
    # Summarize CV results
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Method':<20} {'Accuracy':<15} {'AUC':<15} {'F1':<15}")
    print("-" * 65)
    
    for method, metrics in cv_results.items():
        acc_mean = np.mean(metrics['accuracy'])
        acc_std = np.std(metrics['accuracy'])
        auc_mean = np.mean(metrics['auc'])
        auc_std = np.std(metrics['auc'])
        f1_mean = np.mean(metrics['f1'])
        f1_std = np.std(metrics['f1'])
        
        print(f"{method:<20} {acc_mean:.4f}±{acc_std:.4f} {auc_mean:.4f}±{auc_std:.4f} {f1_mean:.4f}±{f1_std:.4f}")
    
    # Aggregate and analyze SHAP results
    aggregated_shap = aggregate_shap_results(shap_results, X.columns.tolist())
    
    # Create SHAP plots
    create_shap_plots(shap_results, aggregated_shap, max_display=max_display)

    # Aggregate splitting features results
    aggregated_splitting = aggregate_splitting_features(splitting_features_storage, X.columns.tolist())

    # Create aggregated confusion matrices
    print(f"\n{'='*60}")
    print("CREATING AGGREGATED CONFUSION MATRICES")
    print(f"{'='*60}")
    
    create_aggregated_confusion_matrix(cv_results, confusion_matrices)
    
    return cv_results, shap_results, aggregated_shap, best_params_storage, splitting_features_storage, aggregated_splitting, confusion_matrices



from hyperparameter_grid import get_hyperparameter_grid
from nested_cv_util import apply_augmentation, create_classifier, get_classifier_supports_feature_analysis


def nested_cv_with_conditional_analysis(X, y, classifier_name='rf',
                                       augmentation_methods=['none', 'aitchison_mixup', 'feature_dropout', 'cutmix'],
                                       outer_cv=5, inner_cv=3, random_state=42, 
                                       force_cpu=False, max_display=20):
    """
    Perform nested cross-validation with specified classifier and conditional feature analysis
    
    Args:
        classifier_name: 'rf', 'xgb', 'lr', or 'svm'
        Other args same as original function
    """
    print(f"{'='*80}")
    print(f"NESTED CV WITH {classifier_name.upper()} AND CONDITIONAL FEATURE ANALYSIS")
    print(f"{'='*80}")
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Augmentation methods: {augmentation_methods}")
    
    # Check if this classifier supports feature analysis
    supports_feature_analysis = get_classifier_supports_feature_analysis(classifier_name)
    print(f"Feature analysis (SHAP/splitting): {'Enabled' if supports_feature_analysis else 'Disabled'}")
    
    # Initialize results storage
    cv_results = defaultdict(lambda: defaultdict(list))
    shap_results = defaultdict(lambda: defaultdict(dict)) if supports_feature_analysis else None
    best_params_storage = defaultdict(lambda: defaultdict(list))
    confusion_matrices = defaultdict(list)
    splitting_features_storage = defaultdict(dict) if supports_feature_analysis else None
    
    # Initialize SHAP explainer only for supported classifiers
    if supports_feature_analysis:
        shap.initjs()
    
    # Outer CV loop
    outer_skf = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"OUTER FOLD {fold_idx + 1}/{outer_cv}")
        print(f"{'='*60}")
        
        # Split data for this outer fold
        X_train_outer = X.iloc[train_idx]
        X_test_outer = X.iloc[test_idx]
        y_train_outer = y[train_idx]
        y_test_outer = y[test_idx]
        
        print(f"Train size: {len(X_train_outer)}, Test size: {len(X_test_outer)}")
        
        # Test each augmentation method
        for aug_method in augmentation_methods:
            print(f"\n{'-'*40}")
            print(f"AUGMENTATION METHOD: {aug_method}")
            print(f"{'-'*40}")
            
            # Apply augmentation
            if aug_method == 'none':
                X_train_aug = X_train_outer.copy()
                y_train_aug = y_train_outer.copy()
            else:
                X_train_aug, y_train_aug = apply_augmentation(
                    X_train_outer, y_train_outer, method=aug_method, aug_ratio=3.0
                )
            
            print(f"Training samples after augmentation: {len(X_train_aug)}")
            print(f"Class distribution: {np.bincount(y_train_aug)}")
            
            # Inner CV for hyperparameter tuning
            inner_skf = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_state)
            
            # Create base classifier
            base_classifier = create_classifier(classifier_name, params={}, force_cpu=force_cpu)
            
            # Hyperparameter grid
            param_grid = get_hyperparameter_grid(classifier_name)
            
            # Grid search
            print("Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                estimator=base_classifier,
                param_grid=param_grid,
                cv=inner_skf,
                scoring='accuracy',
                n_jobs=-1 if classifier_name != 'svm' else 1,  # SVM can be memory intensive
                verbose=0
            )
            
            # Fit grid search
            grid_search.fit(X_train_aug.values, y_train_aug)
            best_params = grid_search.best_params_
            
            print(f"Best parameters: {best_params}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Store best parameters
            best_params_storage[aug_method][fold_idx] = best_params
            
            # Train final model with best parameters
            final_classifier = create_classifier(classifier_name, params=best_params, force_cpu=force_cpu)
            
            # Handle data preprocessing for different model types
            if 'cuml' in str(type(final_classifier)) and not force_cpu:
                X_train_processed = X_train_aug.values.astype(np.float32)
                X_test_processed = X_test_outer.values.astype(np.float32)
            else:
                X_train_processed = X_train_aug.values
                X_test_processed = X_test_outer.values
            
            # Train final model
            final_classifier.fit(X_train_processed, y_train_aug)
            
            # Evaluate on test set
            y_pred = final_classifier.predict(X_test_processed)
            y_pred_proba = final_classifier.predict_proba(X_test_processed)
            
            # Handle probability output for different model types
            if len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 2:
                y_pred_proba_pos = y_pred_proba[:, 1]
            else:
                y_pred_proba_pos = y_pred_proba
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_outer, y_pred)
            auc = roc_auc_score(y_test_outer, y_pred_proba_pos)
            f1 = f1_score(y_test_outer, y_pred, average='weighted')
            cm, cm_filename = plot_and_save_confusion_matrix(
                y_test_outer, y_pred, f"{classifier_name}_{aug_method}", fold_idx
            )
            confusion_matrices[aug_method].append(cm)
            
            # Store CV results
            cv_results[aug_method]['accuracy'].append(accuracy)
            cv_results[aug_method]['auc'].append(auc)
            cv_results[aug_method]['f1'].append(f1)
            
            print(f"\nPerformance on test set:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC: {auc:.4f}")
            print(f"  F1: {f1:.4f}")
            print(f"  Confusion Matrix:\n{cm}")
            print(f"  Confusion matrix saved: {cm_filename}")
            
            # CONDITIONAL FEATURE ANALYSIS
            if supports_feature_analysis:
                # SHAP Analysis
                print(f"\nPerforming SHAP analysis...")
                try:
                    shap_values, explainer = get_shap_values(
                        final_classifier, X_train_processed, X_test_processed, classifier_name, force_cpu
                    )
                    
                    # Store SHAP results
                    shap_results[aug_method][fold_idx] = {
                        'shap_values': shap_values,
                        'explainer': explainer,
                        'X_test': X_test_processed,
                        'y_test': y_test_outer,
                        'y_pred': y_pred,
                        'feature_names': X.columns.tolist(),
                        'model_type': classifier_name,
                        'aug_method': aug_method,
                        'accuracy': accuracy,
                        'auc': auc,
                        'f1': f1,
                        'best_params': best_params
                    }
                    
                    print(f"SHAP analysis completed successfully")
                    
                except Exception as e:
                    print(f"SHAP analysis failed: {str(e)}")
                    shap_results[aug_method][fold_idx] = {
                        'error': str(e),
                        'feature_names': X.columns.tolist(),
                        'model_type': classifier_name,
                        'aug_method': aug_method,
                        'accuracy': accuracy,
                        'auc': auc,
                        'f1': f1,
                        'best_params': best_params
                    }
                
                # SPLITTING FEATURES ANALYSIS (only for tree-based models)
                if classifier_name in ['rf', 'xgb']:
                    print(f"\nPerforming splitting features analysis...")
                    try:
                        splitting_results = perform_enhanced_splitting_analysis(
                            final_classifier, X_train_processed, y_train_aug, X.columns.tolist(), 
                            f"{classifier_name}_{aug_method}", fold_idx
                        )
                        
                        # Store splitting results
                        splitting_features_storage[aug_method][fold_idx] = splitting_results
                        
                    except Exception as e:
                        print(f"Splitting features analysis failed: {str(e)}")
                        splitting_features_storage[aug_method][fold_idx] = None
            else:
                print(f"\nSkipping feature analysis for {classifier_name} (not supported)")
    
    # Summarize CV results
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION RESULTS SUMMARY - {classifier_name.upper()}")
    print(f"{'='*80}")
    
    print(f"\n{'Method':<20} {'Accuracy':<15} {'AUC':<15} {'F1':<15}")
    print("-" * 65)
    
    for method, metrics in cv_results.items():
        acc_mean = np.mean(metrics['accuracy'])
        acc_std = np.std(metrics['accuracy'])
        auc_mean = np.mean(metrics['auc'])
        auc_std = np.std(metrics['auc'])
        f1_mean = np.mean(metrics['f1'])
        f1_std = np.std(metrics['f1'])
        
        print(f"{method:<20} {acc_mean:.4f}±{acc_std:.4f} {auc_mean:.4f}±{auc_std:.4f} {f1_mean:.4f}±{f1_std:.4f}")
    
    # Conditional feature analysis aggregation and plotting
    aggregated_shap = None
    aggregated_splitting = None
    
    if supports_feature_analysis:
        # Aggregate and analyze SHAP results
        aggregated_shap = aggregate_shap_results(shap_results, X.columns.tolist())
        
        # Create SHAP plots
        create_shap_plots(shap_results, aggregated_shap, max_display=max_display)
        
        # Aggregate splitting features results (only for tree-based)
        if classifier_name in ['rf', 'xgb']:
            aggregated_splitting = aggregate_splitting_features(splitting_features_storage, X.columns.tolist())
    
    # Create aggregated confusion matrices
    print(f"\n{'='*60}")
    print("CREATING AGGREGATED CONFUSION MATRICES")
    print(f"{'='*60}")
    
    create_aggregated_confusion_matrix(cv_results, confusion_matrices)
    
    return (cv_results, shap_results, aggregated_shap, best_params_storage, 
            splitting_features_storage, aggregated_splitting, confusion_matrices)