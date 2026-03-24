import pandas as pd
import numpy as np
from scipy.stats import kruskal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def perform_kruskal_wallis_analysis(merged_df):
    """
    Performs Kruskal-Wallis H-test on OTU counts to identify significant differences
    between ASD and Control groups and filters the dataset to retain only significant OTUs.

    Args:
        merged_df (pd.DataFrame): The merged DataFrame containing OTU counts,
                                  'Status' (ASD/Control), and 'Cohort' information.

    Returns:
        tuple: A tuple containing:
            - X_significant (pd.DataFrame): Feature matrix with only significant OTUs.
            - y (np.array): Labels array.
    """

    autism_samples = merged_df.index[merged_df['Status'] == 'ASD'].tolist()
    control_samples = merged_df.index[merged_df['Status'] == 'Control'].tolist()
    print(f"Autism samples: {len(autism_samples)}")
    print(f"Control samples: {len(control_samples)}")

    features = merged_df.iloc[:, :-8]
    labels = np.array([1 if merged_df.loc[col, 'Status'] == 'ASD' else 0 for col in features.index])

    features_rel = features.div(features.sum(axis=1), axis=0) * 100
    features_log = np.log10(features + 1)
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        index=features.index,
            columns=features.columns
    )

    # Create a feature matrix from the OTU counts
    features = merged_df.iloc[:, :-8]  # Transpose so samples are rows, OTUs are columns
    labels = np.array([1 if merged_df.loc[col, 'Status'] == 'ASD' else 0 for col in features.index])

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
    # KRUSKAL-WALLIS TESTING
    # =====================================
    print("\n" + "="*60)
    print("KRUSKAL-WALLIS TEST RESULTS")
    print("="*60)
    print("Testing each OTU for significant differences between ASD and Control groups")
    print("(Non-parametric test - doesn't assume normal distribution)")
    print()

    # Separate the data by group for Kruskal-Wallis test
    asd_data = X[y == 1]  # ASD samples
    control_data = X[y == 0]  # Control samples

    print(f"ASD group: {len(asd_data)} samples")
    print(f"Control group: {len(control_data)} samples")
    print(f"Testing {X.shape[1]} OTUs")
    print()

    # Store results
    kruskal_results = []
    identical_otus = []  # Track OTUs with identical values

    print("Running Kruskal-Wallis tests...")
    for i, otu in enumerate(X.columns):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(X.columns)} OTUs...")

        # Get the OTU values for each group
        asd_values = asd_data[otu].values
        control_values = control_data[otu].values

        # Skip if either group has too few samples
        if len(asd_values) < 3 or len(control_values) < 3:
            continue

        # Check for identical values (common cause of Kruskal-Wallis errors)
        if len(np.unique(np.concatenate([asd_values, control_values]))) == 1:
            identical_otus.append(otu)
            continue

        # Perform Kruskal-Wallis test
        try:
            statistic, p_value = kruskal(asd_values, control_values)

            # Calculate medians for effect size understanding
            asd_median = np.median(asd_values)
            control_median = np.median(control_values)

            kruskal_results.append({
                'OTU': otu,
                'statistic': statistic,
                'p_value': p_value,
                'asd_median': asd_median,
                'control_median': control_median,
                'fold_change': asd_median / (control_median + 1e-10)  # Avoid division by zero
            })
        except Exception as e:
            print(f"  Error processing OTU {otu}: {e}")
            continue

    print(f"Completed testing {len(kruskal_results)} OTUs")
    print(f"Skipped {len(identical_otus)} OTUs with identical values across all samples")
    print()

    # Convert to DataFrame for easier handling
    kruskal_df = pd.DataFrame(kruskal_results)

    # Use raw p-values without FDR correction
    print("Using raw p-values (no multiple testing correction applied)...")
    if len(kruskal_df) > 0:
        # Sort by p-value
        kruskal_df = kruskal_df.sort_values('p_value')
        
        # Mark significant OTUs based on raw p-value threshold
        kruskal_df['significant'] = kruskal_df['p_value'] < 0.05

        print(f"Total OTUs tested: {len(kruskal_df)}")
        print(f"Significant OTUs (raw p < 0.05): {np.sum(kruskal_df['significant'])}")
        print()

        # Show top significant results
        significant_otus = kruskal_df[kruskal_df['significant']]

        if len(significant_otus) > 0:
            print("TOP 10 MOST SIGNIFICANT OTUs:")
            print("-" * 80)
            for i, (_, row) in enumerate(significant_otus.head(10).iterrows()):
                direction = "Higher in ASD" if row['asd_median'] > row['control_median'] else "Higher in Control"
                print(f"{i+1:2d}. {row['OTU'][:50]:50s} | p={row['p_value']:.2e} | {direction}")
            print()

            # Summary statistics
            print("SUMMARY STATISTICS:")
            print(f"  Most significant p-value: {significant_otus['p_value'].min():.2e}")
            higher_in_asd = np.sum(significant_otus['asd_median'] > significant_otus['control_median'])
            higher_in_control = np.sum(significant_otus['asd_median'] < significant_otus['control_median'])
            print(f"  OTUs higher in ASD: {higher_in_asd}")
            print(f"  OTUs higher in Control: {higher_in_control}")
            print(f"  Average fold change (ASD/Control): {significant_otus['fold_change'].mean():.2f}")

        else:
            print("No OTUs were significant at p < 0.05.")
            print("Top 10 OTUs by raw p-value:")
            print("-" * 80)
            for i, (_, row) in enumerate(kruskal_df.head(10).iterrows()):
                direction = "Higher in ASD" if row['asd_median'] > row['control_median'] else "Higher in Control"
                print(f"{i+1:2d}. {row['OTU'][:50]:50s} | p={row['p_value']:.2e} | {direction}")

    else:
        print("No valid results from Kruskal-Wallis testing!")
        significant_otus = pd.DataFrame() # Ensure it's defined even if no results

    # =====================================
    # FILTER DATASET TO SIGNIFICANT OTUs ONLY
    # =====================================
    print("\n" + "="*60)
    print("FILTERING DATASET TO SIGNIFICANT OTUs")
    print("="*60)

    if len(significant_otus) > 0:
        # Get the list of significant OTU names
        significant_otu_names = significant_otus['OTU'].tolist()

        print(f"Original dataset shape: {X.shape}")
        print(f"Significant OTUs to keep: {len(significant_otu_names)}")

        # Filter the feature matrices to keep only significant OTUs
        X_significant = X[significant_otu_names].copy()
        X_train_significant = X_train[significant_otu_names].copy() if set(significant_otu_names).issubset(X_train.columns) else X_train[[col for col in significant_otu_names if col in X_train.columns]].copy()
        X_test_significant = X_test[significant_otu_names].copy() if set(significant_otu_names).issubset(X_test.columns) else X_test[[col for col in significant_otu_names if col in X_test.columns]].copy()

        # Also filter the normalized versions
        features_rel_significant = features_rel[significant_otu_names].copy() if set(significant_otu_names).issubset(features_rel.columns) else features_rel[[col for col in significant_otu_names if col in features_rel.columns]].copy()
        features_log_significant = features_log[significant_otu_names].copy() if set(significant_otu_names).issubset(features_log.columns) else features_log[[col for col in significant_otu_names if col in features_log.columns]].copy()
        features_scaled_significant = features_scaled[significant_otu_names].copy() if set(significant_otu_names).issubset(features_scaled.columns) else features_scaled[[col for col in significant_otu_names if col in features_scaled.columns]].copy()

        print(f"Filtered dataset shape: {X_significant.shape}")
        print(f"Filtered train set shape: {X_train_significant.shape}")
        print(f"Filtered test set shape: {X_test_significant.shape}")
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
        for i, (_, row) in enumerate(significant_otus.head(20).iterrows()):
            direction = "↑ASD" if row['asd_median'] > row['control_median'] else "↑Control"
            print(f"  {i+1:2d}. {row['OTU']:30s} | p={row['p_value']:.2e} | {direction}")

        if len(significant_otus) > 20:
            print(f"  ... and {len(significant_otus)-20} more")

        print()
        print("VARIABLES CREATED:")
        print("  X_significant          - Main filtered feature matrix")
        print("  X_train_significant    - Filtered training set")
        print("  X_test_significant     - Filtered test set")
        print("  features_rel_significant   - Filtered relative abundance")
        print("  features_log_significant   - Filtered log-transformed")
        print("  features_scaled_significant - Filtered standardized")
        print("  significant_otu_names      - List of significant OTU names")
        print("  kruskal_df                 - Full Kruskal-Wallis results")
        print("  significant_otus           - Only significant results")

    else:
        print("No significant OTUs found - keeping original dataset unchanged")
        X_significant = X.copy()
        X_train_significant = X_train.copy()
        X_test_significant = X_test.copy()
        features_rel_significant = features_rel.copy()
        features_log_significant = features_log.copy()
        features_scaled_significant = features_scaled.copy()
        significant_otu_names = X.columns.tolist()

    print("\n" + "="*60)
    print("KRUSKAL-WALLIS ANALYSIS COMPLETE")
    print("="*60)

    return X_significant, y