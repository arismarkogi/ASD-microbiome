import os
import numpy as np
import matplotlib.pyplot as plt
import shap
import pandas as pd
from find_best_augmentation_method import find_best_augmentation_method
from nested_cv_util import get_classifier_supports_feature_analysis

def create_and_save_all_plots(cv_results, shap_results, aggregated_shap, 
                                         aggregated_splitting, classifier_name, max_display=20):
    """
    Create and save all plots with conditional feature analysis based on classifier type
    """
    print(f"\n{'='*60}")
    print(f"CREATING AND SAVING ALL PLOTS FOR {classifier_name.upper()}")
    print(f"{'='*60}")
    
    # Create plots directory with classifier name
    plots_dir = f"plots_{classifier_name}"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Always create performance plots
    create_and_save_performance_plots(cv_results, plots_dir)
    
    # Conditional feature analysis plots
    supports_feature_analysis = get_classifier_supports_feature_analysis(classifier_name)
    
    if supports_feature_analysis:
        print(f"Creating feature analysis plots for {classifier_name}")
        
        # SHAP Plots (for RF and XGBoost)
        if shap_results and aggregated_shap:
            create_and_save_shap_plots(shap_results, aggregated_shap, plots_dir, max_display)
        
        # Feature Importance Plots (only for tree-based models)
        if classifier_name in ['rf', 'xgb'] and aggregated_splitting:
            create_and_save_feature_importance_plots(aggregated_splitting, plots_dir, max_display)
    else:
        print(f"Skipping feature analysis plots for {classifier_name} (not supported)")
        # Create a note file explaining why no feature plots
        note_path = os.path.join(plots_dir, "feature_analysis_note.txt")
        with open(note_path, 'w') as f:
            f.write(f"Feature analysis plots not created for {classifier_name}\n")
            f.write(f"Reason: {classifier_name} does not support SHAP or tree-based feature importance\n")
            f.write(f"Only performance and confusion matrix plots are available.\n")
    
    print(f"All plots saved in: {plots_dir}")


def create_and_save_performance_plots(cv_results, plots_dir):
    """
    Create and save performance comparison plots
    """
    # Create performance subdirectory
    perf_plots_dir = os.path.join(plots_dir, "performance_plots")
    os.makedirs(perf_plots_dir, exist_ok=True)
    
    # Performance comparison plot
    methods = list(cv_results.keys())
    metrics = ['accuracy', 'auc', 'f1']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        means = [np.mean(cv_results[method][metric]) for method in methods]
        stds = [np.std(cv_results[method][metric]) for method in methods]
        
        bars = axes[i].bar(methods, means, yerr=stds, capsize=5)
        axes[i].set_ylabel(metric.upper())
        axes[i].set_title(f'{metric.upper()} Comparison')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01, 
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(perf_plots_dir, 'performance_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance plots saved in: {perf_plots_dir}")

def create_and_save_feature_importance_plots(aggregated_splitting, plots_dir, max_display=20):
    """
    Create and save Random Forest built-in feature importance plots
    """
    # Create feature importance subdirectory
    fi_plots_dir = os.path.join(plots_dir, "feature_importance_plots")
    os.makedirs(fi_plots_dir, exist_ok=True)
    
    # 1. Comparison across augmentation methods
    fig, axes = plt.subplots(1, len(aggregated_splitting), figsize=(15, 8))
    if len(aggregated_splitting) == 1:
        axes = [axes]
    
    for idx, (aug_method, agg_data) in enumerate(aggregated_splitting.items()):
        if agg_data is not None:
            summary_df = agg_data['summary_df'].head(max_display)
            
            if 'avg_split_count' in summary_df.columns:
                bars = axes[idx].barh(range(len(summary_df)), summary_df['avg_split_count'],
                                     xerr=summary_df['std_split_count'], capsize=3)
                axes[idx].set_xlabel('Average Split Count')
                title_suffix = f"Split Count - {agg_data['n_folds']} folds"
            else:
                bars = axes[idx].barh(range(len(summary_df)), summary_df['avg_importance'],
                                     xerr=summary_df['std_importance'], capsize=3)
                axes[idx].set_xlabel('Average Importance')
                title_suffix = f"Importance - {agg_data['n_folds']} folds"
            
            axes[idx].set_yticks(range(len(summary_df)))
            axes[idx].set_yticklabels(summary_df['feature'])
            axes[idx].set_title(f'{aug_method}\n{title_suffix}')
            axes[idx].invert_yaxis()
        else:
            axes[idx].text(0.5, 0.5, f'No results\nfor {aug_method}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{aug_method}\n(No results)')
    
    plt.tight_layout()
    plt.suptitle('RF Feature Importance Across Augmentation Methods', y=1.02)
    plt.savefig(os.path.join(fi_plots_dir, 'rf_importance_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Individual plots for each augmentation method
    for aug_method, agg_data in aggregated_splitting.items():
        if agg_data is not None:
            summary_df = agg_data['summary_df'].head(max_display)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if 'avg_split_count' in summary_df.columns:
                bars = ax.barh(range(len(summary_df)), summary_df['avg_split_count'],
                              xerr=summary_df['std_split_count'], capsize=3)
                ax.set_xlabel('Average Split Count ± Std')
                title = f'RF Split Count - {aug_method}\n(Averaged across {agg_data["n_folds"]} folds)'
                
                # Add value labels
                for i, (bar, val, std) in enumerate(zip(bars, summary_df['avg_split_count'], summary_df['std_split_count'])):
                    ax.text(bar.get_width() + std + 1, bar.get_y() + bar.get_height()/2, 
                           f'{val:.1f}', ha='left', va='center', fontsize=8)
            else:
                bars = ax.barh(range(len(summary_df)), summary_df['avg_importance'],
                              xerr=summary_df['std_importance'], capsize=3)
                ax.set_xlabel('Average Importance ± Std')
                title = f'RF Feature Importance - {aug_method}\n(Averaged across {agg_data["n_folds"]} folds)'
                
                # Add value labels
                for i, (bar, val, std) in enumerate(zip(bars, summary_df['avg_importance'], summary_df['std_importance'])):
                    ax.text(bar.get_width() + std + 0.001, bar.get_y() + bar.get_height()/2, 
                           f'{val:.3f}', ha='left', va='center', fontsize=8)
            
            ax.set_yticks(range(len(summary_df)))
            ax.set_yticklabels(summary_df['feature'])
            ax.set_title(title)
            ax.invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(os.path.join(fi_plots_dir, f'rf_importance_{aug_method}.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Feature importance plots saved in: {fi_plots_dir}")

def create_averaged_shap_values(shap_results):
    """
    Create averaged SHAP values and corresponding X_test data across all folds for each method
    """
    averaged_shap_data = {}
    
    for aug_method, method_results in shap_results.items():
        if method_results:  # Check if there are results for this method
            all_shap_values = []
            all_x_test = []
            feature_names = None
            
            # Collect SHAP values and X_test from all folds
            for fold_idx, fold_data in method_results.items():
                if 'shap_values' in fold_data and 'error' not in fold_data:
                    shap_values = fold_data['shap_values']
                    X_test = fold_data['X_test']
                    
                    if feature_names is None:
                        feature_names = fold_data['feature_names']
                    
                    all_shap_values.append(shap_values)
                    all_x_test.append(X_test)
            
            # Concatenate all folds
            if all_shap_values:
                combined_shap_values = np.vstack(all_shap_values)
                combined_x_test = np.vstack(all_x_test)
                
                # Calculate mean and std of SHAP values for each feature
                mean_shap_values = np.mean(combined_shap_values, axis=0)
                std_shap_values = np.std(combined_shap_values, axis=0)
                
                # Store the averaged data
                averaged_shap_data[aug_method] = {
                    'mean_shap_values': mean_shap_values,
                    'std_shap_values': std_shap_values,
                    'combined_shap_values': combined_shap_values,
                    'combined_x_test': combined_x_test,
                    'feature_names': feature_names,
                    'n_samples': len(combined_shap_values),
                    'n_folds': len(all_shap_values)
                }
    
    return averaged_shap_data

def create_fancy_plots_for_fold(shap_values, X_test_df, feature_names, aug_method, fold_idx, method_dir, max_display=20):
    """
    Create fancy SHAP plots for a specific fold of a specific augmentation method
    """
    try:
        # 1. BEESWARM PLOT (the colorful one!)
        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(shap.Explanation(values=shap_values, 
                                           data=X_test_df.values, 
                                           feature_names=feature_names),
                          max_display=max_display, show=False)
        plt.title(f'SHAP Beeswarm Plot - {aug_method} - Fold {fold_idx + 1}')
        plt.savefig(os.path.join(method_dir, f'beeswarm_fold_{fold_idx+1}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. WATERFALL PLOTS for first few predictions
        base_value = np.mean(shap_values.sum(axis=1))  # More robust base value calculation
        for i in range(min(3, len(X_test_df))):  # First 3 predictions
            plt.figure(figsize=(12, 8))
            shap.plots.waterfall(shap.Explanation(values=shap_values[i], 
                                                base_values=base_value,
                                                data=X_test_df.iloc[i].values,
                                                feature_names=feature_names),
                               max_display=max_display, show=False)
            plt.title(f'SHAP Waterfall - {aug_method} - Fold {fold_idx+1} - Prediction {i+1}')
            plt.savefig(os.path.join(method_dir, f'waterfall_fold_{fold_idx+1}_pred_{i+1}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. HEATMAP PLOT (for first 50 samples to keep it readable)
        plt.figure(figsize=(12, 8))
        n_samples = min(50, len(shap_values))
        shap.plots.heatmap(shap.Explanation(values=shap_values[:n_samples], 
                                          data=X_test_df.iloc[:n_samples].values,
                                          feature_names=feature_names),
                         max_display=max_display, show=False)
        plt.title(f'SHAP Heatmap - {aug_method} - Fold {fold_idx + 1} (First {n_samples} samples)')
        plt.savefig(os.path.join(method_dir, f'heatmap_fold_{fold_idx+1}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. VIOLIN PLOT
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_df, plot_type="violin",
                        max_display=max_display, show=False)
        plt.title(f'SHAP Violin Plot - {aug_method} - Fold {fold_idx + 1}')
        plt.savefig(os.path.join(method_dir, f'violin_fold_{fold_idx+1}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. PARTIAL DEPENDENCE PLOTS for top features (if dataset is not too large)
        if len(X_test_df) <= 1000:  # Only for smaller datasets to avoid performance issues
            # Fixed: Get top features using numpy operations directly on SHAP values
            mean_abs_shap = np.abs(shap_values).mean(axis=0)  # Mean absolute SHAP values per feature
            top_feature_indices = np.argsort(mean_abs_shap)[-min(5, max_display):]  # Top N features
            
            for feature_idx in top_feature_indices:
                feature_name = feature_names[feature_idx]
                try:
                    plt.figure(figsize=(10, 6))
                    
                    # Create a simple partial dependence plot manually
                    feature_values = X_test_df.iloc[:, feature_idx].values
                    shap_values_for_feature = shap_values[:, feature_idx]
                    
                    # Sort by feature values for smooth curve
                    sorted_indices = np.argsort(feature_values)
                    sorted_feature_values = feature_values[sorted_indices]
                    sorted_shap_values = shap_values_for_feature[sorted_indices]
                    
                    # Create scatter plot with trend line
                    plt.scatter(sorted_feature_values, sorted_shap_values, alpha=0.6, s=30)
                    
                    # Add smooth trend line using rolling average
                    if len(sorted_feature_values) > 10:
                        window_size = max(10, len(sorted_feature_values) // 20)
                        rolling_shap = pd.Series(sorted_shap_values).rolling(window=window_size, center=True).mean()
                        plt.plot(sorted_feature_values, rolling_shap, color='red', linewidth=2, label='Trend')
                        plt.legend()
                    
                    plt.xlabel(f'Feature Value: {feature_name}')
                    plt.ylabel('SHAP Value')
                    plt.title(f'Feature Effect - {aug_method} - Fold {fold_idx + 1}\nFeature: {feature_name}')
                    plt.grid(True, alpha=0.3)
                    
                    plt.savefig(os.path.join(method_dir, f'feature_effect_{feature_name.replace(" ", "_")}_fold_{fold_idx+1}.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    print(f"Could not create feature effect plot for {feature_name}: {str(e)}")
                    plt.close()  # Ensure plot is closed even if there's an error
                
    except Exception as e:
        print(f"Could not create fancy SHAP plots for {aug_method} fold {fold_idx + 1}: {str(e)}")
        # Make sure any open plots are closed
        plt.close('all')

def create_averaged_beeswarm_plots(averaged_shap_data, shap_plots_dir, max_display=20):
    """
    Create beeswarm plots using averaged SHAP values across folds
    """
    avg_plots_dir = os.path.join(shap_plots_dir, "averaged_plots")
    os.makedirs(avg_plots_dir, exist_ok=True)
    
    # 1. Individual averaged beeswarm plots for each method
    for aug_method, avg_data in averaged_shap_data.items():
        try:
            # Create DataFrame for easier handling
            X_test_df = pd.DataFrame(avg_data['combined_x_test'], 
                                   columns=avg_data['feature_names'])
            shap_values = avg_data['combined_shap_values']
            
            # Beeswarm plot with all combined data
            plt.figure(figsize=(12, 8))
            shap.plots.beeswarm(shap.Explanation(values=shap_values, 
                                               data=X_test_df.values, 
                                               feature_names=avg_data['feature_names']),
                              max_display=max_display, show=False)
            plt.title(f'Averaged SHAP Beeswarm Plot - {aug_method}\n'
                     f'(Combined data from {avg_data["n_folds"]} folds, '
                     f'{avg_data["n_samples"]} total samples)')
            plt.savefig(os.path.join(avg_plots_dir, f'averaged_beeswarm_{aug_method}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Summary plot (bar version) with averaged data
            plt.figure(figsize=(12, 6))
            shap.summary_plot(shap_values, X_test_df, plot_type="bar",
                            max_display=max_display, show=False)
            plt.title(f'Averaged SHAP Summary (Bar) - {aug_method}\n'
                     f'(Combined data from {avg_data["n_folds"]} folds)')
            plt.savefig(os.path.join(avg_plots_dir, f'averaged_summary_bar_{aug_method}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Violin plot with averaged data
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test_df, plot_type="violin",
                            max_display=max_display, show=False)
            plt.title(f'Averaged SHAP Violin Plot - {aug_method}\n'
                     f'(Combined data from {avg_data["n_folds"]} folds)')
            plt.savefig(os.path.join(avg_plots_dir, f'averaged_violin_{aug_method}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Could not create averaged beeswarm plot for {aug_method}: {str(e)}")
    
    # 2. Comparison plot across methods using representative samples
    if len(averaged_shap_data) > 1:
        create_method_comparison_beeswarm(averaged_shap_data, avg_plots_dir, max_display)
    
    print(f"Averaged SHAP plots saved in: {avg_plots_dir}")

def create_method_comparison_beeswarm(averaged_shap_data, avg_plots_dir, max_display=20, samples_per_method=200):
    """
    Create side-by-side beeswarm plots for method comparison
    """
    methods = list(averaged_shap_data.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 8))
    if n_methods == 1:
        axes = [axes]
    
    for idx, (aug_method, avg_data) in enumerate(averaged_shap_data.items()):
        try:
            # Subsample data for comparison (to keep plots readable)
            n_samples = min(samples_per_method, avg_data['n_samples'])
            indices = np.random.choice(avg_data['n_samples'], n_samples, replace=False)
            
            sampled_shap = avg_data['combined_shap_values'][indices]
            sampled_x = avg_data['combined_x_test'][indices]
            
            X_test_df = pd.DataFrame(sampled_x, columns=avg_data['feature_names'])
            
            # Create beeswarm on specific axis
            plt.sca(axes[idx])  # Set current axis
            shap.plots.beeswarm(shap.Explanation(values=sampled_shap, 
                                               data=X_test_df.values, 
                                               feature_names=avg_data['feature_names']),
                              max_display=max_display, show=False)
            axes[idx].set_title(f'{aug_method}\n({n_samples} samples)')
            
        except Exception as e:
            print(f"Could not create comparison beeswarm for {aug_method}: {str(e)}")
            axes[idx].text(0.5, 0.5, f'Error creating\nplot for {aug_method}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
    
    plt.suptitle('SHAP Beeswarm Comparison Across Augmentation Methods', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(avg_plots_dir, 'method_comparison_beeswarm.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_and_save_shap_plots(shap_results, aggregated_shap, plots_dir, max_display=20):
    """
    Create and save comprehensive SHAP visualization plots for all augmentation methods
    """
    plt.style.use('default')
    
    # Create SHAP subdirectory
    shap_plots_dir = os.path.join(plots_dir, "shap_plots")
    os.makedirs(shap_plots_dir, exist_ok=True)
    
    # Create averaged SHAP values for beeswarm plots
    averaged_shap_data = create_averaged_shap_values(shap_results)
    
    # Create averaged beeswarm plots
    if averaged_shap_data:
        create_averaged_beeswarm_plots(averaged_shap_data, shap_plots_dir, max_display)
    
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
    plt.savefig(os.path.join(shap_plots_dir, 'shap_comparison_across_methods.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mean SHAP plots for each augmentation method (aggregated across folds)
    for aug_method, agg_data in aggregated_shap.items():
        if agg_data is not None:
            # Create mean SHAP importance plot
            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = agg_data['feature_importance_df'].head(max_display)
            
            bars = ax.barh(range(len(top_features)), top_features['importance'],
                          xerr=top_features['importance_std'], capsize=3)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Mean |SHAP Value| ± Std')
            ax.set_title(f'Mean SHAP Feature Importance - {aug_method}\n(Averaged across {agg_data["n_folds"]} folds)')
            ax.invert_yaxis()
            
            # Add value labels on bars
            for i, (bar, val, std) in enumerate(zip(bars, top_features['importance'], top_features['importance_std'])):
                ax.text(bar.get_width() + std + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', ha='left', va='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(shap_plots_dir, f'mean_shap_{aug_method}.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    # 3. Individual fold SHAP plots for ALL augmentation methods
    for aug_method, method_results in shap_results.items():
        if method_results:  # Check if there are results for this method
            method_dir = os.path.join(shap_plots_dir, f"individual_folds_{aug_method}")
            os.makedirs(method_dir, exist_ok=True)
            
            print(f"Creating individual fold SHAP plots for method: {aug_method}")
            
            for fold_idx, fold_data in method_results.items():
                if 'shap_values' in fold_data and 'error' not in fold_data:
                    try:
                        shap_values = fold_data['shap_values']
                        X_test = fold_data['X_test']
                        feature_names = fold_data['feature_names']
                        
                        if len(shap_values.shape) == 2:
                            X_test_df = pd.DataFrame(X_test, columns=feature_names)
                            
                            # Basic SHAP summary plot (dot plot)
                            plt.figure(figsize=(12, 8))
                            shap.summary_plot(shap_values, X_test_df,
                                            max_display=max_display, show=False)
                            plt.title(f'SHAP Summary - {aug_method} - Fold {fold_idx + 1}')
                            plt.savefig(os.path.join(method_dir, f'shap_summary_fold_{fold_idx+1}.png'), 
                                       dpi=300, bbox_inches='tight')
                            plt.close()
                            
                            # SHAP bar plot
                            plt.figure(figsize=(12, 6))
                            shap.summary_plot(shap_values, X_test_df, plot_type="bar",
                                            max_display=max_display, show=False)
                            plt.title(f'SHAP Bar Plot - {aug_method} - Fold {fold_idx + 1}')
                            plt.savefig(os.path.join(method_dir, f'shap_bar_fold_{fold_idx+1}.png'), 
                                       dpi=300, bbox_inches='tight')
                            plt.close()
                            
                            # Create fancy plots for each method
                            create_fancy_plots_for_fold(shap_values, X_test_df, feature_names, 
                                                       aug_method, fold_idx, method_dir, max_display)
                            
                    except Exception as e:
                        print(f"Could not create SHAP plots for {aug_method} fold {fold_idx + 1}: {str(e)}")
    
    # Create method comparison plots
    create_method_comparison_plots(shap_results, aggregated_shap, shap_plots_dir, max_display)
    
    print(f"SHAP plots saved in: {shap_plots_dir}")




def create_method_comparison_plots(shap_results, aggregated_shap, shap_plots_dir, max_display=20):
    """
    Create additional comparison plots across all methods
    """
    comparison_dir = os.path.join(shap_plots_dir, "method_comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 1. Top features consistency across methods
    all_features = set()
    method_feature_importance = {}
    
    for aug_method, agg_data in aggregated_shap.items():
        if agg_data is not None:
            features_df = agg_data['feature_importance_df']
            method_feature_importance[aug_method] = dict(zip(features_df['feature'], features_df['importance']))
            all_features.update(features_df['feature'].tolist())
    
    if len(method_feature_importance) > 1 and all_features:
        # Create heatmap of feature importance across methods
        feature_matrix = []
        feature_names = sorted(list(all_features))
        methods = list(method_feature_importance.keys())
        
        for feature in feature_names:
            row = []
            for method in methods:
                importance = method_feature_importance[method].get(feature, 0)
                row.append(importance)
            feature_matrix.append(row)
        
        # Plot top N features only
        top_n = min(max_display, len(feature_names))
        avg_importance = [np.mean(row) for row in feature_matrix]
        top_indices = np.argsort(avg_importance)[-top_n:]
        
        plt.figure(figsize=(12, 8))
        heatmap_data = np.array([feature_matrix[i] for i in top_indices])
        top_feature_names = [feature_names[i] for i in top_indices]
        
        im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='SHAP Importance')
        plt.yticks(range(len(top_feature_names)), top_feature_names)
        plt.xticks(range(len(methods)), methods, rotation=45)
        plt.title(f'Feature Importance Heatmap Across Methods\n(Top {top_n} features)')
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'feature_importance_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Method comparison plots saved in: {comparison_dir}")