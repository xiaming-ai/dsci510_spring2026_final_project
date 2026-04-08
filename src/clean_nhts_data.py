import pandas as pd
import os

def main():
    data_dir = os.path.join('..', '..', 'NHTS 2017 copy')
    output_csv = os.path.join('..', 'data', 'combined_nhts_data.csv')
    
    # Read the datasets 
    hhpub = pd.read_csv(os.path.join(data_dir, 'hhpub.csv'), low_memory=False)
    perpub = pd.read_csv(os.path.join(data_dir, 'perpub.csv'), low_memory=False)
    vehpub = pd.read_csv(os.path.join(data_dir, 'vehpub.csv'), low_memory=False)
    trippub = pd.read_csv(os.path.join(data_dir, 'trippub.csv'), low_memory=False)
    df_combined = trippub.copy()
    
    def get_cols_to_merge(df_left, df_right, keys):
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

    numeric_cols = df_combined.select_dtypes(include=['number']).columns
    
    df_combined[numeric_cols] = df_combined[numeric_cols].fillna(df_combined[numeric_cols].mean())
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    df_combined.to_csv(output_csv, index=False)

if __name__ == "__main__":
    main()
