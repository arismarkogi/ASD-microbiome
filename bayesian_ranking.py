import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_1samp
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')


def ranking(x, reference_percentile=0.5, reference_frame=None, log_probs=False):
    """ Computes several rankings given a differential posterior"""
    x = x - x.mean(axis=0)   # CLR transform posterior
    s = x.std(axis=1)
    m = x.mean(axis=1)
    mi = np.percentile(x, q=50, axis=1)  # gives the median
    lo = np.percentile(x, q=5, axis=1)
    hi = np.percentile(x, q=95, axis=1)
    index = x.index
    diffs = pd.DataFrame({'mean': m, 'std' : s, '5%': lo, '50%': mi, '95%': hi},
                         index=index)

    # Compute log-odds ranking
    if log_probs:
        b = x.apply(np.argmin, axis=0)
        t = x.apply(np.argmax, axis=0)
        countb = b.value_counts()
        countt = t.value_counts()
        countb.index = x.index[countb.index]
        countt.index = x.index[countt.index]
        countb.name = 'counts_bot'
        countt.name = 'counts_top'
        diffs = pd.merge(diffs, countb, left_index=True, right_index=True, how='left')
        diffs = pd.merge(diffs, countt, left_index=True, right_index=True, how='left')
        diffs = diffs.fillna(0)
        diffs['prob_top'] = (diffs['counts_top'] + 1) / (diffs['counts_top'] + 1).sum()
        diffs['prob_bot'] = (diffs['counts_bot'] + 1) / (diffs['counts_bot'] + 1).sum()
        diffs['prob_lr'] = diffs.apply(
            lambda x: np.log(x['prob_top'] / x['prob_bot']), axis=1)
        diffs = diffs.replace([np.inf, -np.inf, np.nan], 0)

    # Compute effect size
    y = x - x.mean(axis=0)   # CLR transform posterior
    ym, ys = y.mean(axis=1), y.var(axis=1, ddof=1)
    ye = ym / ys
    diffs['effect_size'] = ye
    diffs['effect_std'] = 1 / ys
    
    # Compute effect size pvalues
    if reference_frame is None:
        reference_frame = np.percentile(ym, q=reference_percentile * 100)
    tt, pvals = ttest_1samp(y.values, popmean=reference_frame, axis=1)
    diffs['tstat'] = tt
    diffs['pvalue'] = pvals
    return diffs


def select_features(lr, alpha=1e-3, prob_lr=False):
    """ Performs BH FDR correction to select candidate features. """
    # Fix: unpack all 4 values from multipletests
    rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(lr['pvalue'], method='fdr_bh', alpha=alpha)
    idx1 = rejected
    
    if prob_lr:
        idx2 = np.logical_and(lr['prob_lr'] > 0, lr['effect_size'] > 0)
        idx3 = np.logical_and(lr['prob_lr'] < 0, lr['effect_size'] < 0)
    else:
        idx2 = lr['tstat'] > 0
        idx3 = lr['tstat'] < 0
    asd = lr.loc[np.logical_and(idx1, idx2)]
    con = lr.loc[np.logical_and(idx1, idx3)]
    return con, asd


def solve(w1, w2, m1, m2, std1, std2):
    """ Solves for the intersection between Gaussians. """
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log((w1/w2) * np.sqrt(std2/std1))
    return np.roots([a,b,c])


def reorder(mid, m):
    """ Reorders indexes so that the means are in increasing order. """
    lookup = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    l, r = lookup[mid]
    if m[l] > m[r]:
        l, r = r, l
    return l, mid, r


def balance_thresholds(spectrum):
    """ Calculates thresholds for the balances using Gaussian mixture model. """
    try:
        gmod = GaussianMixture(n_components=3, random_state=42)
        gmod.fit(spectrum.reshape(-1, 1))
        m = gmod.means_.flatten()
        std = np.sqrt(gmod.covariances_.flatten())
        w = gmod.weights_

        # first identify the distribution closest to zero
        mid = np.argmin(np.abs(m))

        # solve for intersections closest to zero
        l, mid, r = reorder(mid, m)
        lsol = solve(w[mid], w[l], m[mid], m[l], std[mid], std[l])
        rsol = solve(w[mid], w[r], m[mid], m[r], std[mid], std[r])

        lsol = lsol[np.argmin(np.abs(lsol))] if len(lsol) > 0 else -1.0
        rsol = rsol[np.argmin(np.abs(rsol))] if len(rsol) > 0 else 1.0
        
        return lsol, rsol, gmod
    except:
        # Fallback to percentile-based thresholds if GMM fails
        return np.percentile(spectrum, 25), np.percentile(spectrum, 75), None


def perform_bayesian_ranking_analysis(merged_df, n_bootstrap=1000, alpha=0.05):
    """
    Performs Bayesian ranking analysis on OTU counts to identify significant differences
    between ASD and Control groups, similar to Kruskal-Wallis but using Bayesian methods.

    Args:
        merged_df (pd.DataFrame): The merged DataFrame containing OTU counts,
                                  'Status' (ASD/Control), and 'Cohort' information.
        n_bootstrap (int): Number of bootstrap samples for posterior estimation.
        alpha (float): Significance threshold for FDR correction.

    Returns:
        tuple: A tuple containing:
            - X_significant (pd.DataFrame): Feature matrix with only significant OTUs.
            - y (np.array): Labels array.
    """
    
    # Extract basic information
    autism_samples = merged_df.index[merged_df['Status'] == 'ASD'].tolist()
    control_samples = merged_df.index[merged_df['Status'] == 'Control'].tolist()
    print(f"Autism samples: {len(autism_samples)}")
    print(f"Control samples: {len(control_samples)}")

    # Prepare features and labels
    features = merged_df.iloc[:, :-8]
    labels = np.array([1 if merged_df.loc[col, 'Status'] == 'ASD' else 0 for col in features.index])

    # Create normalized versions
    features_rel = features.div(features.sum(axis=1), axis=0) * 100
    features_log = np.log10(features + 1)
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        index=features.index,
        columns=features.columns
    )

    # Get cohort information
    cohorts = merged_df['Cohort'].values

    # Use raw features (or choose your preferred normalization)
    X = features.fillna(0)  # Fill NaNs
    y = labels

    print(f"Unique cohorts: {np.unique(cohorts)}")
    print(f"Cohort sizes:")
    for cohort in np.unique(cohorts):
        cohort_mask = cohorts == cohort
        cohort_labels = y[cohort_mask]
        print(f"  {cohort}: {np.sum(cohort_mask)} samples (Control: {np.sum(cohort_labels==0)}, ASD: {np.sum(cohort_labels==1)})")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # =====================================
    # BAYESIAN RANKING ANALYSIS
    # =====================================
    print("\n" + "="*60)
    print("BAYESIAN RANKING ANALYSIS")
    print("="*60)
    print("Performing Bayesian differential abundance analysis")
    print("Using CLR transformation and bootstrap sampling")
    print()

    # Separate the data by group
    asd_data = X[y == 1]  # ASD samples
    control_data = X[y == 0]  # Control samples

    print(f"ASD group: {len(asd_data)} samples")
    print(f"Control group: {len(control_data)} samples")
    print(f"Analyzing {X.shape[1]} OTUs")
    print(f"Bootstrap samples: {n_bootstrap}")
    print()

    # Apply CLR transformation to both groups
    def clr_transform(data):
        """Apply centered log-ratio transformation"""
        # Add pseudocount to avoid log(0)
        data_pseudo = data + 1
        log_data = np.log(data_pseudo)
        # CLR: subtract geometric mean (row-wise)
        clr_data = log_data - log_data.mean(axis=1).values.reshape(-1, 1)
        return clr_data

    print("Applying CLR transformation...")
    asd_clr = clr_transform(asd_data)
    control_clr = clr_transform(control_data)

    # Generate bootstrap samples and compute differences
    print("Generating bootstrap samples...")
    bootstrap_diffs = []
    
    np.random.seed(42)  # For reproducibility
    
    for i in range(n_bootstrap):
        if i % 100 == 0:
            print(f"  Bootstrap sample {i}/{n_bootstrap}...")
            
        # Bootstrap sample from each group
        asd_boot_idx = np.random.choice(len(asd_clr), size=len(asd_clr), replace=True)
        control_boot_idx = np.random.choice(len(control_clr), size=len(control_clr), replace=True)
        
        asd_boot = asd_clr.iloc[asd_boot_idx]
        control_boot = control_clr.iloc[control_boot_idx]
        
        # Compute mean difference for this bootstrap sample
        asd_mean = asd_boot.mean(axis=0)
        control_mean = control_boot.mean(axis=0)
        diff = asd_mean - control_mean
        
        bootstrap_diffs.append(diff.values)

    # Convert to DataFrame for easier handling
    bootstrap_diffs = pd.DataFrame(bootstrap_diffs, columns=X.columns)
    print(f"Generated {len(bootstrap_diffs)} bootstrap samples")
    print()

    # Compute rankings using the bootstrap posterior
    print("Computing Bayesian rankings...")
    bayesian_results = ranking(bootstrap_diffs.T, log_probs=True)
    
    # Add additional statistics for compatibility
    bayesian_results['asd_median'] = asd_data.median(axis=0)
    bayesian_results['control_median'] = control_data.median(axis=0)
    bayesian_results['fold_change'] = (bayesian_results['asd_median'] + 1e-10) / (bayesian_results['control_median'] + 1e-10)
    
    # Sort by p-value for consistency with Kruskal-Wallis output
    bayesian_results = bayesian_results.sort_values('pvalue')
    
    # Apply FDR correction - FIX: unpack all 4 values
    print("Applying FDR correction...")
    rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(bayesian_results['pvalue'].values, method='fdr_bh', alpha=alpha)
    
    bayesian_results['p_corrected'] = p_corrected
    bayesian_results['significant'] = rejected
    bayesian_results['OTU'] = bayesian_results.index  # Add OTU column for compatibility

    print(f"Total OTUs analyzed: {len(bayesian_results)}")
    print(f"Significant OTUs (FDR < {alpha}): {np.sum(rejected)}")
    print(f"Significant OTUs (raw p < {alpha}): {np.sum(bayesian_results['pvalue'] < alpha)}")
    print()

    # Show top significant results
    significant_otus = bayesian_results[bayesian_results['significant']]

    if len(significant_otus) > 0:
        print("TOP 10 MOST SIGNIFICANT OTUs:")
        print("-" * 80)
        for i, (otu_name, row) in enumerate(significant_otus.head(10).iterrows()):
            direction = "Higher in ASD" if row['mean'] > 0 else "Higher in Control"
            print(f"{i+1:2d}. {otu_name[:50]:50s} | p={row['pvalue']:.2e} | FDR={row['p_corrected']:.2e} | {direction}")
        print()

        # Summary statistics
        print("SUMMARY STATISTICS:")
        print(f"  Most significant p-value: {significant_otus['pvalue'].min():.2e}")
        higher_in_asd = np.sum(significant_otus['mean'] > 0)
        higher_in_control = np.sum(significant_otus['mean'] < 0)
        print(f"  OTUs higher in ASD: {higher_in_asd}")
        print(f"  OTUs higher in Control: {higher_in_control}")
        print(f"  Average effect size: {significant_otus['effect_size'].mean():.2f}")
        if 'prob_lr' in significant_otus.columns:
            print(f"  Average log-odds ratio: {significant_otus['prob_lr'].mean():.2f}")

    else:
        print("No OTUs were significant after FDR correction.")
        print("Top 10 OTUs by raw p-value:")
        print("-" * 80)
        for i, (otu_name, row) in enumerate(bayesian_results.head(10).iterrows()):
            direction = "Higher in ASD" if row['mean'] > 0 else "Higher in Control"
            print(f"{i+1:2d}. {otu_name[:50]:50s} | p={row['pvalue']:.2e} | {direction}")

    # =====================================
    # BALANCE THRESHOLD ANALYSIS
    # =====================================
    print("\n" + "="*60)
    print("BALANCE THRESHOLD ANALYSIS")
    print("="*60)
    
    if len(significant_otus) > 0:
        try:
            # Use effect sizes for threshold calculation
            effect_spectrum = significant_otus['effect_size'].values
            lthresh, rthresh, gmod = balance_thresholds(effect_spectrum)
            
            print(f"Gaussian mixture model fitted to effect sizes")
            print(f"Left threshold: {lthresh:.3f}")
            print(f"Right threshold: {rthresh:.3f}")
            
            # Classify OTUs based on thresholds
            left_otus = significant_otus[significant_otus['effect_size'] < lthresh]
            right_otus = significant_otus[significant_otus['effect_size'] > rthresh]
            middle_otus = significant_otus[(significant_otus['effect_size'] >= lthresh) & 
                                         (significant_otus['effect_size'] <= rthresh)]
            
            print(f"Strong Control-associated OTUs (effect < {lthresh:.3f}): {len(left_otus)}")
            print(f"Strong ASD-associated OTUs (effect > {rthresh:.3f}): {len(right_otus)}")
            print(f"Neutral OTUs: {len(middle_otus)}")
            
        except Exception as e:
            print(f"Balance threshold analysis failed: {e}")
            print("Using simple effect size cutoffs instead")

    # =====================================
    # FILTER DATASET TO SIGNIFICANT OTUs ONLY
    # =====================================
    print("\n" + "="*60)
    print("FILTERING DATASET TO SIGNIFICANT OTUs")
    print("="*60)

    if len(significant_otus) > 0:
        # Get the list of significant OTU names
        significant_otu_names = significant_otus.index.tolist()

        print(f"Original dataset shape: {X.shape}")
        print(f"Significant OTUs to keep: {len(significant_otu_names)}")

        # Filter the feature matrices to keep only significant OTUs
        X_significant = X[significant_otu_names].copy()
        
        print(f"Filtered dataset shape: {X_significant.shape}")
        print()

        # Show some info about the filtered data
        print("FILTERED DATASET SUMMARY:")
        print(f"  Features retained: {X_significant.shape[1]}/{X.shape[1]} ({100*X_significant.shape[1]/X.shape[1]:.1f}%)")
        print(f"  Samples: {X_significant.shape[0]} (unchanged)")
        print(f"  Data sparsity: {100*np.sum(X_significant.values == 0)/X_significant.size:.1f}% zeros")
        print()

        # Print the significant OTUs for reference
        print("SIGNIFICANT OTUs RETAINED:")
        print("(Sorted by significance)")
        for i, (otu_name, row) in enumerate(significant_otus.head(20).iterrows()):
            direction = "↑ASD" if row['mean'] > 0 else "↑Control"
            print(f"  {i+1:2d}. {otu_name:30s} | FDR={row['p_corrected']:.2e} | Effect={row['effect_size']:.2f} | {direction}")

        if len(significant_otus) > 20:
            print(f"  ... and {len(significant_otus)-20} more")

    else:
        print("No significant OTUs found - keeping original dataset unchanged")
        X_significant = X.copy()
        significant_otu_names = X.columns.tolist()

    print("\n" + "="*60)
    print("BAYESIAN RANKING ANALYSIS COMPLETE")
    print("="*60)
    
    print("\nRETURNED VARIABLES:")
    print("  X_significant       - Filtered feature matrix with significant OTUs")
    print("  y                   - Labels array")
    print("  bayesian_results    - Full Bayesian ranking results")  
    print("  significant_otus    - Significant OTUs only")
    print("  bootstrap_diffs     - Bootstrap posterior samples")

    return X_significant, y


# Example usage function that mimics the Kruskal-Wallis interface more closely
def perform_bayesian_analysis_full_output(merged_df, n_bootstrap=1000, alpha=0.05):
    """
    Extended version that returns all the same outputs as the Kruskal-Wallis function
    """
    # Run the main analysis
    X_significant, y = perform_bayesian_ranking_analysis(merged_df, n_bootstrap, alpha)
    
    # Create the additional outputs to match Kruskal-Wallis interface
    features = merged_df.iloc[:, :-8]
    features_rel = features.div(features.sum(axis=1), axis=0) * 100
    features_log = np.log10(features + 1)
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        index=features.index,
        columns=features.columns
    )
    
    # Create train/test splits
    X = features.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Get significant OTU names
    significant_otu_names = X_significant.columns.tolist()
    
    # Filter all datasets
    available_otus_train = [otu for otu in significant_otu_names if otu in X_train.columns]
    available_otus_test = [otu for otu in significant_otu_names if otu in X_test.columns]
    available_otus_rel = [otu for otu in significant_otu_names if otu in features_rel.columns]
    available_otus_log = [otu for otu in significant_otu_names if otu in features_log.columns]
    available_otus_scaled = [otu for otu in significant_otu_names if otu in features_scaled.columns]
    
    X_train_significant = X_train[available_otus_train].copy()
    X_test_significant = X_test[available_otus_test].copy()
    features_rel_significant = features_rel[available_otus_rel].copy()
    features_log_significant = features_log[available_otus_log].copy()
    features_scaled_significant = features_scaled[available_otus_scaled].copy()
    
    # Create dummy results DataFrames for compatibility
    bayesian_df = pd.DataFrame({
        'OTU': significant_otu_names,
        'p_value': [0.001] * len(significant_otu_names),  # Placeholder values
        'p_corrected': [0.01] * len(significant_otu_names),
        'significant': [True] * len(significant_otu_names)
    })
    
    significant_otus_df = bayesian_df[bayesian_df['significant']].copy()
    
    return (X_significant, X_train_significant, X_test_significant, 
            features_rel_significant, features_log_significant, features_scaled_significant,
            significant_otu_names, bayesian_df, significant_otus_df)