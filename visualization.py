import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualizations():
    try:
        X = pd.read_csv('x.csv')
        y = pd.read_csv('y.csv')
    except FileNotFoundError:
        print("Error")
        return

    target_col = y.columns[0]
    df_numeric = X.select_dtypes(include=['number']).copy()
    df_numeric['TARGET_MODE'] = y[target_col]
    corr_matrix = df_numeric.corr()
    
    plt.figure(figsize=(14, 12))
    
     
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, vmin=-1, vmax=1)
    
    plt.title('Feature Correlation Matrix', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    corr_filename = 'correlation_map.png'
    plt.savefig(corr_filename, dpi=300)
    print(f"Saved correlation map to [{corr_filename}]")
    plt.close()

 
    df_viz = pd.DataFrame()
    
    if 'CENSUS_R' in X.columns:
        df_viz['CENSUS_R'] = X['CENSUS_R']
    else:
        print("Error")
        return
        
    df_viz['Travel_Mode'] = y[target_col]

    focus_modes = {
        1: 'Walking (01)', 
        3: 'Car (03)', 
        11: 'Public Transit (11)'
    }
    

    df_filtered = df_viz[df_viz['Travel_Mode'].isin([1, 3, 11])].copy()
    

    df_filtered['Travel_Mode_Label'] = df_filtered['Travel_Mode'].map(focus_modes)
    census_map = {1: 'Northeast', 2: 'Midwest', 3: 'South', 4: 'West'}
    df_filtered['CENSUS_R_Label'] = df_filtered['CENSUS_R'].map(census_map)
    grouped = df_filtered.groupby(['CENSUS_R_Label', 'Travel_Mode_Label']).size().unstack(fill_value=0)

    if grouped.empty:
         return
    ax = grouped.plot(kind='bar', figsize=(14, 8), width=0.8, colormap='viridis')
    
    ### AI generated with plot part
    plt.title('Total Travel Mode Count per Census Area (CENSUS_R)', fontsize=18, fontweight='bold')
    plt.xlabel('Census Region (CENSUS_R)', fontsize=14, fontweight='bold')
    plt.ylabel('Total Count of Trips', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0, fontsize=12) 
    plt.yticks(fontsize=12)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Travel Mode', title_fontsize=13, fontsize=12)
    
    plt.tight_layout()
    bar_filename = 'travel_modes_by_area.png'
    plt.savefig(bar_filename, dpi=300)
    plt.close()
    ###

if __name__ == "__main__":
    create_visualizations()
