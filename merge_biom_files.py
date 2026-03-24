
import pandas as pd
from biom import load_table

def merge_biom_tables(biom_file1, biom_file2, output_file):
    """
    Merges two BIOM tables and writes the result to a new file.

    Parameters:
    - biom_file1: str, path to the first BIOM file
    - biom_file2: str, path to the second BIOM file
    - output_file: str, path where the merged BIOM file will be saved
    """
    table1 = load_table(biom_file1)
    table2 = load_table(biom_file2)

    print(f"Table 1: {table1.shape[0]} OTUs x {table1.shape[1]} samples")
    print(f"Table 2: {table2.shape[0]} OTUs x {table2.shape[1]} samples")

    # Merge the tables
    merged_table = table1.merge(table2)

    print(f"Merged Table: {merged_table.shape[0]} OTUs x {merged_table.shape[1]} samples")

    # Save the merged table to a new BIOM file
    with open(output_file, 'w') as f:
        merged_table.to_json("Merged via biom", f)

    print(f"Merged BIOM file saved to: {output_file}")



def read_biom_with_taxonomy(biom_file_path):
    """
    Reads a .biom file and prints OTU IDs, optionally including taxonomy annotations.
    """
    table = load_table(biom_file_path)
    sample_ids = table.ids(axis='sample')
    otu_ids = table.ids(axis='observation')

    print(f"Found {len(otu_ids)} OTUs and {len(sample_ids)} samples.\n")


    print("Sample IDs:", table.ids(axis='sample'))

    # Observation (row) IDs = similar to .var index (OTU or ASV IDs)
    print("Observation IDs:", table.ids(axis='observation'))

    # Sample metadata = similar to .obs
    print("Sample metadata:", table.metadata(axis='sample'))

    # Observation metadata = similar to .var (taxonomy, etc.)
    print("Observation metadata:", table.metadata(axis='observation'))

def biom_to_df(table):
    """
    Convert a BIOM table to a pandas DataFrame (samples x OTUs).
    """
    df = pd.DataFrame(
        table.matrix_data.toarray().T,  # transpose so rows = samples
        index=table.ids(axis='sample'),
        columns=table.ids(axis='observation')
    )
    return df


def load_metadata(metadata_fp):
    """
    Load metadata file as a DataFrame with sample_id as index.
    """
    return pd.read_table(metadata_fp, sep='\t', engine='python').set_index('sampleid')

def merge_biom_tables_with_metadata(biom_fp1, biom_fp2, meta_fp1, meta_fp2):
    # Load BIOM tables
    table1 = load_table(biom_fp1)
    table2 = load_table(biom_fp2)

    df1 = biom_to_df(table1)
    df2 = biom_to_df(table2)

    # Combine OTU tables
    otu_merged = pd.concat([df1, df2], axis=0)

    # Fill missing values with 0, preserving index and column names
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    otu_merged_imputed_array = imputer.fit_transform(otu_merged)
    otu_merged_imputed = pd.DataFrame(otu_merged_imputed_array,
                                      index=otu_merged.index,
                                      columns=otu_merged.columns)

    # Load metadata and merge
    meta1 = load_metadata(meta_fp1)
    meta2 = load_metadata(meta_fp2)
    meta_merged = pd.concat([meta1, meta2], axis=0)

    # Ensure metadata and OTU data have matching indices
    # Keep only samples present in both metadata and OTU data
    common_samples = otu_merged_imputed.index.intersection(meta_merged.index)
    otu_merged_imputed = otu_merged_imputed.loc[common_samples]
    meta_merged = meta_merged.loc[common_samples]


    print("Harmonization completed!")
    print(f"Shape: {otu_merged_imputed.shape}")

    # Join harmonized OTU data with full metadata
    merged_df = otu_merged_imputed.join(meta_merged, how='inner')

    # Identify MJ features to remove
    mj_features = [col for col in merged_df.columns if col.startswith("MJ")]

    # Drop them from the dataset
    merged_df = merged_df.drop(columns=mj_features)

    print(f"Final merged shape: {merged_df.shape}")

    return merged_df

