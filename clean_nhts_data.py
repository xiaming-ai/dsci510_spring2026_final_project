import pandas as pd
import os

def main():
    # Define paths
    # Using parent directory to point to 'NHTS 2017 copy'
    data_dir = os.path.join('..', '..', 'NHTS 2017 copy')
    output_csv = os.path.join('..', 'data', 'combined_nhts_data.csv')
    
    print("Loading datasets...")
    # Read the datasets (low_memory=False to avoid DtypeWarnings on mixed columns)
    hhpub = pd.read_csv(os.path.join(data_dir, 'hhpub.csv'), low_memory=False)
    perpub = pd.read_csv(os.path.join(data_dir, 'perpub.csv'), low_memory=False)
    vehpub = pd.read_csv(os.path.join(data_dir, 'vehpub.csv'), low_memory=False)
    trippub = pd.read_csv(os.path.join(data_dir, 'trippub.csv'), low_memory=False)
    
    print("Combining data...")
    # trippub is the most granular and already contains many columns from others.
    # We will do left joins to bring in any missing columns from perpub, vehpub, and hhpub.
    df_combined = trippub.copy()
    
    def get_cols_to_merge(df_left, df_right, keys):
        # Only bring in columns that do not already exist in the left dataframe (to avoid _x and _y suffix duplicates)
        cols_to_keep = keys + [c for c in df_right.columns if c not in df_left.columns]
        return df_right[cols_to_keep]

    # 1. Merge person data
    per_merge_cols = get_cols_to_merge(df_combined, perpub, ['HOUSEID', 'PERSONID'])
    df_combined = df_combined.merge(per_merge_cols, on=['HOUSEID', 'PERSONID'], how='left')

    # 2. Merge vehicle data
    veh_merge_cols = get_cols_to_merge(df_combined, vehpub, ['HOUSEID', 'VEHID'])
    df_combined = df_combined.merge(veh_merge_cols, on=['HOUSEID', 'VEHID'], how='left')

    # 3. Merge household data
    hh_merge_cols = get_cols_to_merge(df_combined, hhpub, ['HOUSEID'])
    df_combined = df_combined.merge(hh_merge_cols, on=['HOUSEID'], how='left')

    # IMPORTANT NOTE: In the NHTS dataset, missing values are typically coded as negative numbers 
    # (e.g. -1: Appropriate skip, -7: Refused, -8: Don't know, -9: Not ascertained).
    # If you want these negative codes treated as missing (NaN) before imputing the mean, 
    # you can uncomment the following lines:
    # 
    # import numpy as np
    # numeric_cols = df_combined.select_dtypes(include=['number']).columns
    # df_combined[numeric_cols] = df_combined[numeric_cols].applymap(lambda x: np.nan if x < 0 else x)

    print("Replacing missing values with mean...")
    # Select numeric columns to avoid attempting to average strings/objects
    numeric_cols = df_combined.select_dtypes(include=['number']).columns
    
    # Fill NaN values with the mean of each respective numeric column
    df_combined[numeric_cols] = df_combined[numeric_cols].fillna(df_combined[numeric_cols].mean())
    
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    print(f"Saving combined data to {output_csv}...")
    df_combined.to_csv(output_csv, index=False)
    print("Done. Data cleaning complete!")

if __name__ == "__main__":
    main()
