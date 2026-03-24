from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.inspection import permutation_importance


def perform_enhanced_splitting_analysis(model, X_train, y_train, feature_names, aug_method, fold_idx):
    """
    Perform enhanced splitting analysis with better cuML support
    """
    print(f"\n{'='*60}")
    print(f"ENHANCED SPLITTING ANALYSIS - {aug_method} - Fold {fold_idx + 1}")
    print(f"{'='*60}")
    
    try:
        # Perform enhanced analysis
        splitting_results = get_splitting_features_analysis(
            model, X_train, y_train, feature_names, top_n=10
        )
        
        if splitting_results[0] is not None:
            print_enhanced_splitting_summary(splitting_results, aug_method, fold_idx)
            
            # Print insights
            insights = get_splitting_insights(splitting_results)
            if insights:
                print(f"\n  💡 Key Insights:")
                print(f"    • Most used feature: {insights.get('most_used_feature', 'N/A')}")
                if 'most_used_feature_splits' in insights:
                    print(f"    • Used in {insights['most_used_feature_splits']} splits "
                          f"({insights['most_used_feature_frequency']:.1%} of total)")
                if 'top_5_concentration' in insights:
                    print(f"    • Top 5 features account for {insights['top_5_concentration']:.1%} of splits")
            
            return splitting_results
        else:
            
            print(f"  ❌ Analysis failed - no results returned")
            return None
            
    except Exception as e:
        print(f"  ❌ Enhanced splitting analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
def get_splitting_features_analysis(model, X_train, y_train, feature_names, top_n=10):
    """
    Enhanced analysis for splitting features that works with both sklearn and cuML models
    Uses multiple approaches to determine which features are most important for splitting
    """
    results = {}
    
    # Approach 1: Feature importances (when available)
    feature_importances = None
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        results['method_1'] = 'sklearn_feature_importances'
    elif hasattr(model, 'get_feature_importance'):
        # Some models have this method
        try:
            feature_importances = model.get_feature_importance()
            results['method_1'] = 'custom_feature_importances'
        except:
            pass
    
    # Approach 2: Train equivalent sklearn model to get splitting info
    sklearn_model = None
    if 'cuml' in str(type(model)) or feature_importances is None:
        print("  Training equivalent sklearn model for splitting analysis...")
        try:
            # Get model parameters
            params = {}
            if hasattr(model, 'n_estimators'):
                params['n_estimators'] = model.n_estimators
            if hasattr(model, 'max_depth'):
                params['max_depth'] = model.max_depth
            if hasattr(model, 'min_samples_split'):
                params['min_samples_split'] = model.min_samples_split
            if hasattr(model, 'min_samples_leaf'):
                params['min_samples_leaf'] = model.min_samples_leaf
            if hasattr(model, 'random_state'):
                params['random_state'] = model.random_state
            
            # Create and train sklearn equivalent
            sklearn_model = RandomForestClassifier(
                n_jobs=-1,
                **params
            )
            sklearn_model.fit(X_train, y_train)
            
            # Get feature importances
            if feature_importances is None:
                feature_importances = sklearn_model.feature_importances_
                results['method_1'] = 'sklearn_equivalent_importances'
            
            results['sklearn_model'] = sklearn_model
            
        except Exception as e:
            print(f"  Failed to create sklearn equivalent: {str(e)}")
    
    # Approach 3: Get detailed splitting information from sklearn model
    split_counts = None
    split_frequencies = None
    total_splits = 0
    
    analysis_model = sklearn_model if sklearn_model is not None else model
    
    if hasattr(analysis_model, 'estimators_') and analysis_model.estimators_:
        try:
            print(f"  Analyzing {len(analysis_model.estimators_)} trees for splitting patterns...")
            
            # Count splits per feature across all trees
            split_counts = np.zeros(len(feature_names))
            depth_weighted_splits = np.zeros(len(feature_names))
            feature_split_depths = defaultdict(list)
            
            for tree_idx, estimator in enumerate(analysis_model.estimators_):
                if hasattr(estimator, 'tree_'):
                    tree = estimator.tree_
                    
                    # Get tree structure
                    feature_indices = tree.feature
                    left_children = tree.children_left
                    right_children = tree.children_right
                    
                    # Calculate depth for each node
                    def get_node_depths(node_id, current_depth=0):
                        depths = {node_id: current_depth}
                        if left_children[node_id] != -1:  # Has left child
                            depths.update(get_node_depths(left_children[node_id], current_depth + 1))
                        if right_children[node_id] != -1:  # Has right child
                            depths.update(get_node_depths(right_children[node_id], current_depth + 1))
                        return depths
                    
                    node_depths = get_node_depths(0)
                    
                    # Count splits and track depths
                    for node_id, feature_idx in enumerate(feature_indices):
                        if feature_idx >= 0:  # Valid feature (not leaf node)
                            split_counts[feature_idx] += 1
                            total_splits += 1
                            
                            # Weight by inverse depth (earlier splits are more important)
                            depth = node_depths.get(node_id, 0)
                            depth_weight = 1.0 / (depth + 1)
                            depth_weighted_splits[feature_idx] += depth_weight
                            
                            # Track split depths for analysis
                            feature_split_depths[feature_idx].append(depth)
            
            print(f"  Total splits analyzed: {total_splits}")
            
            if total_splits > 0:
                split_frequencies = split_counts / total_splits
                depth_weighted_frequencies = depth_weighted_splits / np.sum(depth_weighted_splits)
                
                # Calculate average split depth for each feature
                avg_split_depths = {}
                for feat_idx, depths in feature_split_depths.items():
                    avg_split_depths[feat_idx] = np.mean(depths)
                
                results.update({
                    'split_counts': split_counts,
                    'split_frequencies': split_frequencies,
                    'depth_weighted_splits': depth_weighted_splits,
                    'depth_weighted_frequencies': depth_weighted_frequencies,
                    'avg_split_depths': avg_split_depths,
                    'total_splits': total_splits,
                    'method_2': 'detailed_tree_analysis'
                })
                
        except Exception as e:
            print(f"  Error in detailed tree analysis: {str(e)}")
    
    # Approach 4: Permutation importance as fallback
    if feature_importances is None:
        try:
            print("  Computing permutation importance...")
            perm_importance = permutation_importance(
                model, X_train, y_train, n_repeats=10, random_state=42
            )
            feature_importances = perm_importance.importances_mean
            results['method_1'] = 'permutation_importance'
        except Exception as e:
            print(f"  Permutation importance failed: {str(e)}")
    
    # Create results DataFrame
    if feature_importances is not None:
        results_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        })
        
        # Add splitting information if available
        if split_counts is not None:
            results_df['split_count'] = split_counts
            results_df['split_frequency'] = split_frequencies
            results_df['depth_weighted_splits'] = depth_weighted_splits
            results_df['depth_weighted_frequency'] = depth_weighted_frequencies
            
            # Add average split depth
            results_df['avg_split_depth'] = [
                avg_split_depths.get(i, np.nan) for i in range(len(feature_names))
            ]
            
            # Sort by split count (primary) and importance (secondary)
            results_df = results_df.sort_values(
                ['split_count', 'importance'], 
                ascending=[False, False]
            )
        else:
            # Sort by importance only
            results_df = results_df.sort_values('importance', ascending=False)
        
        return results_df.head(top_n), results
    
    return None, results

def print_enhanced_splitting_summary(splitting_results, aug_method, fold_idx):
    """
    Print enhanced summary of splitting features analysis
    """
    if splitting_results is None or splitting_results[0] is None:
        print(f"  ❌ No splitting feature analysis available for {aug_method} - Fold {fold_idx + 1}")
        return
    
    results_df, analysis_info = splitting_results
    
    print(f"\n  🌳 ENHANCED SPLITTING ANALYSIS - {aug_method} - Fold {fold_idx + 1}")
    print(f"  {'='*70}")
    
    # Print methods used
    methods_used = []
    if 'method_1' in analysis_info:
        methods_used.append(f"Importance: {analysis_info['method_1']}")
    if 'method_2' in analysis_info:
        methods_used.append(f"Splitting: {analysis_info['method_2']}")
    
    print(f"  Methods used: {', '.join(methods_used)}")
    
    # Print main results
    if 'split_counts' in analysis_info:
        print(f"  Total splits analyzed: {analysis_info['total_splits']:,}")
        print(f"\n  Top {len(results_df)} Most Frequent Splitting Features:")
        print(f"  {'Rank':<4} {'Feature':<25} {'Splits':<8} {'Freq':<8} {'Depth':<8} {'Imp':<8}")
        print(f"  {'-'*65}")
        
        for i, (_, row) in enumerate(results_df.iterrows()):
            splits = int(row['split_count']) if 'split_count' in row else 0
            freq = row['split_frequency'] if 'split_frequency' in row else 0
            depth = row['avg_split_depth'] if 'avg_split_depth' in row else 0
            imp = row['importance']
            
            print(f"  {i+1:<4} {row['feature']:<25} {splits:<8} {freq:<8.3f} {depth:<8.1f} {imp:<8.3f}")
    
    else:
        print(f"  Feature importance analysis only (no splitting details)")
        print(f"\n  Top {len(results_df)} Features by Importance:")
        print(f"  {'Rank':<4} {'Feature':<25} {'Importance':<12}")
        print(f"  {'-'*45}")
        
        for i, (_, row) in enumerate(results_df.iterrows()):
            print(f"  {i+1:<4} {row['feature']:<25} {row['importance']:<12.4f}")

def get_splitting_insights(splitting_results):
    """
    Extract key insights from splitting analysis
    """
    if splitting_results is None or splitting_results[0] is None:
        return None
    
    results_df, analysis_info = splitting_results
    insights = {}
    
    # Top splitting features
    insights['top_splitting_features'] = results_df.head(5)['feature'].tolist()
    
    if 'split_counts' in analysis_info:
        # Splitting concentration
        split_counts = analysis_info['split_counts']
        total_splits = analysis_info['total_splits']
        
        # Calculate how concentrated the splits are
        top_5_splits = np.sum(np.sort(split_counts)[-5:])
        concentration_ratio = top_5_splits / total_splits if total_splits > 0 else 0
        
        insights.update({
            'total_splits': total_splits,
            'top_5_concentration': concentration_ratio,
            'most_used_feature': results_df.iloc[0]['feature'],
            'most_used_feature_splits': int(results_df.iloc[0]['split_count']),
            'most_used_feature_frequency': results_df.iloc[0]['split_frequency']
        })
        
        # Depth analysis
        if 'avg_split_depths' in analysis_info:
            avg_depths = analysis_info['avg_split_depths']
            shallow_features = [
                feat for feat, depth in avg_depths.items() 
                if depth < 2.0 and analysis_info['split_counts'][feat] > 0
            ]
            insights['shallow_splitting_features'] = [
                results_df.iloc[i]['feature'] for i in range(len(results_df))
                if i in shallow_features
            ][:5]
    
    return insights

def aggregate_splitting_features(all_splitting_results, feature_names):
    """
    Enhanced aggregation of splitting features analysis
    """
    aggregated_splitting = {}
    
    for aug_method, fold_results in all_splitting_results.items():
        if not fold_results:
            continue
            
        # Collect results from all folds
        method_importances = []
        method_split_counts = []
        method_split_frequencies = []
        method_insights = []
        successful_folds = 0
        
        for fold_idx, splitting_results in fold_results.items():
            if splitting_results is not None and splitting_results[0] is not None:
                results_df, analysis_info = splitting_results
                
                # Reorder to match original feature order
                ordered_results = pd.DataFrame({'feature': feature_names})
                ordered_results = ordered_results.merge(results_df, on='feature', how='left').fillna(0)
                
                method_importances.append(ordered_results['importance'].values)
                
                if 'split_counts' in analysis_info:
                    method_split_counts.append(ordered_results['split_count'].values)
                    method_split_frequencies.append(ordered_results['split_frequency'].values)
                
                # Get insights
                insights = get_splitting_insights(splitting_results)
                if insights:
                    method_insights.append(insights)
                
                successful_folds += 1
        
        if method_importances:
            # Calculate averages across folds
            avg_importance = np.mean(method_importances, axis=0)
            std_importance = np.std(method_importances, axis=0)
            
            aggregated_data = {
                'feature_names': feature_names,
                'avg_importance': avg_importance,
                'std_importance': std_importance,
                'n_folds': successful_folds
            }
            
            if method_split_counts:
                avg_split_counts = np.mean(method_split_counts, axis=0)
                std_split_counts = np.std(method_split_counts, axis=0)
                avg_split_frequencies = np.mean(method_split_frequencies, axis=0)
                std_split_frequencies = np.std(method_split_frequencies, axis=0)
                
                aggregated_data.update({
                    'avg_split_counts': avg_split_counts,
                    'std_split_counts': std_split_counts,
                    'avg_split_frequencies': avg_split_frequencies,
                    'std_split_frequencies': std_split_frequencies
                })
            
            # Aggregate insights
            if method_insights:
                # Most frequently mentioned features across folds
                all_top_features = []
                for insight in method_insights:
                    all_top_features.extend(insight.get('top_splitting_features', []))
                
                from collections import Counter
                top_features_count = Counter(all_top_features)
                aggregated_data['consistently_top_features'] = [
                    feat for feat, count in top_features_count.most_common(10)
                ]
                
                # Average metrics
                if any('total_splits' in insight for insight in method_insights):
                    avg_total_splits = np.mean([
                        insight.get('total_splits', 0) for insight in method_insights
                    ])
                    aggregated_data['avg_total_splits'] = avg_total_splits
            
            # Create summary DataFrame
            summary_df = pd.DataFrame({
                'feature': feature_names,
                'avg_importance': avg_importance,
                'std_importance': std_importance
            })
            
            if method_split_counts:
                summary_df['avg_split_count'] = avg_split_counts
                summary_df['std_split_count'] = std_split_counts
                summary_df['avg_split_frequency'] = avg_split_frequencies
                summary_df['std_split_frequency'] = std_split_frequencies
                summary_df = summary_df.sort_values('avg_split_count', ascending=False)
            else:
                summary_df = summary_df.sort_values('avg_importance', ascending=False)
            
            aggregated_data['summary_df'] = summary_df
            aggregated_splitting[aug_method] = aggregated_data
    
    return aggregated_splitting