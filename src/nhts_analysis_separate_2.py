
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
print("Loading data...")
HH_COLS   = ['HOUSEID','HHVEHCNT','HHFAMINC','URBRUR','HHSIZE',
             'LIF_CYC','HOMEOWN','YOUNGCHILD','CNTTDHH',
             'BUS','RAIL','WALK','URBANSIZE','HH_RACE']
PER_COLS  = ['HOUSEID','PERSONID','R_AGE','EDUC','R_SEX',
             'MEDCOND','PHYACT','TIMETOWK','OCCAT','WRK_HOME',
             'WRKTRANS','HHFAMINC']
TRIP_COLS = ['HOUSEID','PERSONID','TRPTRANS','TRPMILES','TRVLCMIN','WHYTO']
hh   = pd.read_csv('hhpub.csv',   usecols=HH_COLS,   low_memory=False)
per  = pd.read_csv('perpub.csv',  usecols=PER_COLS,  low_memory=False)
trip = pd.read_csv('trippub.csv', usecols=TRIP_COLS, low_memory=False)
merged = per.merge(hh, on='HOUSEID', how='left')
print(f"  Household records : {hh.shape[0]:,}")
print(f"  Person records    : {per.shape[0]:,}")
print(f"  Trip records      : {trip.shape[0]:,}")
print("Data loaded successfully.\n")
def clean(df, col, valid):
    """Keep only rows whose value in col is in valid."""
    return df[df[col].isin(valid)].copy()
def save(filename):
    """Apply tight layout, save figure, then close it."""
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")
PALETTE = sns.color_palette("Set2")
sns.set_theme(style='whitegrid', font_scale=1.0)
inc_labels = {
    1: '<$10K', 2: '$10-15K', 3: '$15-25K', 4: '$25-35K',
    5: '$35-50K', 6: '$50-75K', 7: '$75-100K', 8: '$100-125K',
    9: '$125-150K', 10: '$150-200K', 11: '>$200K'
}
mode_labels = {
    1:'Walk', 3:'Car', 4:'SUV', 5:'Van', 6:'Pickup',
    10:'School Bus', 11:'Public Bus', 15:'Commuter Rail',
    16:'Subway/LRT', 17:'Taxi/TNC', 19:'Airplane'
}
# Q1. Vehicle ownership rate by income level
print("Q1: Vehicle ownership rate by income level...")
df_q1 = clean(hh, 'HHFAMINC', list(inc_labels.keys())).copy()
df_q1['has_vehicle'] = (df_q1['HHVEHCNT'] > 0).astype(int)
rate = df_q1.groupby('HHFAMINC')['has_vehicle'].mean() * 100
labels = [inc_labels[i] for i in rate.index]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(rate)), rate.values, color=PALETTE[0], edgecolor='white')
ax.set_xticks(range(len(rate)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('Vehicle Ownership Rate (%)')
ax.set_xlabel('Household Income Level')
ax.set_title('Q1. Vehicle Ownership Rate by Income Level', fontweight='bold', fontsize=13)
ax.set_ylim(0, 110)
for bar, val in zip(bars, rate.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1,
            f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
save('q01_vehicle_by_income.png')
# Q2. Regular public transit use rate by income level
print("Q2: Regular public transit use rate by income level...")
df_q2 = clean(hh, 'HHFAMINC', list(inc_labels.keys())).copy()
df_q2 = clean(df_q2, 'BUS', [1, 2, 3, 4, 5])
df_q2['uses_bus'] = (df_q2['BUS'].isin([1, 2])).astype(int)
rate2 = df_q2.groupby('HHFAMINC')['uses_bus'].mean() * 100
labels2 = [inc_labels[i] for i in rate2.index]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(len(rate2)), rate2.values, marker='o', color=PALETTE[1],
        linewidth=2.5, markersize=7)
ax.fill_between(range(len(rate2)), rate2.values, alpha=0.15, color=PALETTE[1])
ax.set_xticks(range(len(rate2)))
ax.set_xticklabels(labels2, rotation=45, ha='right')
ax.set_ylabel('Regular Transit Use Rate (%)\n(Daily or a few times per week)')
ax.set_xlabel('Household Income Level')
ax.set_title('Q2. Regular Public Transit Use Rate by Income Level',
             fontweight='bold', fontsize=13)
for x, val in enumerate(rate2.values):
    ax.text(x, val + 0.2, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
save('q02_transit_by_income.png')
# Q3. Daily trips per household: urban vs rural
print("Q3: Daily trips per household — urban vs rural...")
df_q3 = clean(hh, 'URBRUR', [1, 2]).copy()
df_q3 = df_q3[df_q3['CNTTDHH'] >= 0]
data_urban = df_q3[df_q3['URBRUR'] == 1]['CNTTDHH']
data_rural = df_q3[df_q3['URBRUR'] == 2]['CNTTDHH']
fig, ax = plt.subplots(figsize=(8, 6))
bp = ax.boxplot([data_urban, data_rural], labels=['Urban', 'Rural'],
                patch_artist=True, showfliers=False, widths=0.5)
for patch, color in zip(bp['boxes'], [PALETTE[2], PALETTE[3]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Daily Trips per Household')
ax.set_xlabel('Area Type')
ax.set_title('Q3. Daily Trips per Household: Urban vs Rural',
             fontweight='bold', fontsize=13)
ax.text(1, data_urban.mean() + 0.15, f'Mean: {data_urban.mean():.2f}',
        ha='center', fontsize=10, color='#333')
ax.text(2, data_rural.mean() + 0.15, f'Mean: {data_rural.mean():.2f}',
        ha='center', fontsize=10, color='#333')
save('q03_trips_urban_rural.png')
# Q4. Regular walking rate by age group
print("Q4: Regular walking rate by age group...")
df_q4 = merged.copy()
df_q4 = df_q4[(df_q4['R_AGE'] >= 0) & (df_q4['WALK'].isin([1,2,3,4,5]))]
df_q4['age_group'] = pd.cut(df_q4['R_AGE'],
                             bins=[0,17,29,44,59,74,100],
                             labels=['<18','18-29','30-44','45-59','60-74','75+'])
df_q4['walks_regularly'] = (df_q4['WALK'].isin([1, 2])).astype(int)
walk_rate = df_q4.groupby('age_group', observed=True)['walks_regularly'].mean() * 100
fig, ax = plt.subplots(figsize=(9, 6))
bars4 = ax.bar(walk_rate.index.astype(str), walk_rate.values,
               color=PALETTE[4], edgecolor='white', width=0.6)
ax.set_ylabel('Regular Walking Rate (%)\n(Daily or a few times per week)')
ax.set_xlabel('Age Group')
ax.set_title('Q4. Regular Walking Rate by Age Group', fontweight='bold', fontsize=13)
for bar, val in zip(bars4, walk_rate.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
save('q04_walking_by_age.png')
# Q5. Trip mode share: households with vs without young children
print("Q5: Trip mode share — households with vs without young children...")
df_q5 = trip.merge(hh[['HOUSEID','YOUNGCHILD']], on='HOUSEID', how='left')
df_q5 = df_q5[df_q5['YOUNGCHILD'] >= 0]
df_q5 = clean(df_q5, 'TRPTRANS', list(mode_labels.keys()))
df_q5['has_child'] = (df_q5['YOUNGCHILD'] > 0).map(
    {True: 'Has Young Child', False: 'No Young Child'})
df_q5['mode_label'] = df_q5['TRPTRANS'].map(mode_labels)
ct = df_q5.groupby(['has_child','mode_label']).size().unstack(fill_value=0)
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(12, 6))
ct_pct.T.plot(kind='bar', ax=ax, color=[PALETTE[0], PALETTE[5]],
              edgecolor='white', width=0.7)
ax.set_ylabel('Share of Trips (%)')
ax.set_xlabel('Trip Mode')
ax.set_title('Q5. Trip Mode Share: Households With vs Without Young Children',
             fontweight='bold', fontsize=13)
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Household Type', fontsize=10)
save('q05_mode_by_children.png')
# Q6. Average vehicles per household by urbanization level
print("Q6: Average vehicles per household by urbanization level...")
urbansize_labels = {
    1: 'Large Urban\n(>1M)', 2: 'Mid Urban\n(250K-1M)',
    3: 'Small Urban\n(<250K)', 4: 'Rural'
}
df_q6 = clean(hh, 'URBANSIZE', list(urbansize_labels.keys()))
avg_veh = df_q6.groupby('URBANSIZE')['HHVEHCNT'].mean()
labels6 = [urbansize_labels[i] for i in avg_veh.index]
fig, ax = plt.subplots(figsize=(8, 6))
bars6 = ax.bar(labels6, avg_veh.values,
               color=[PALETTE[i % len(PALETTE)] for i in range(len(avg_veh))],
               edgecolor='white', width=0.5)
ax.set_ylabel('Average Vehicles per Household')
ax.set_xlabel('Urbanization Level')
ax.set_title('Q6. Average Vehicles per Household by Urbanization Level',
             fontweight='bold', fontsize=13)
for bar, val in zip(bars6, avg_veh.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=11)
save('q06_vehicles_by_urbansize.png')
# Q7. Vehicle ownership rate: homeowner vs renter
print("Q7: Vehicle ownership rate — homeowner vs renter...")
own_labels = {1: 'Own / Buying', 2: 'Renting', 97: 'Other'}
df_q7 = clean(hh, 'HOMEOWN', [1, 2, 97]).copy()
df_q7['has_vehicle'] = (df_q7['HHVEHCNT'] > 0).astype(int)
df_q7['own_label'] = df_q7['HOMEOWN'].map(own_labels)
rate7 = df_q7.groupby('own_label')['has_vehicle'].mean() * 100
fig, ax = plt.subplots(figsize=(7, 6))
bars7 = ax.bar(rate7.index, rate7.values,
               color=[PALETTE[0], PALETTE[1], PALETTE[2]],
               edgecolor='white', width=0.45)
ax.set_ylabel('Vehicle Ownership Rate (%)')
ax.set_xlabel('Housing Status')
ax.set_title('Q7. Vehicle Ownership Rate: Owner vs Renter',
             fontweight='bold', fontsize=13)
ax.set_ylim(0, 110)
for bar, val in zip(bars7, rate7.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
save('q07_vehicle_owner_vs_renter.png')
# Q8. Trip mode share: with vs without medical condition
print("Q8: Trip mode share — with vs without medical condition...")
df_q8 = trip.merge(per[['HOUSEID','PERSONID','MEDCOND']],
                   on=['HOUSEID','PERSONID'], how='left')
df_q8 = clean(df_q8, 'MEDCOND', [1, 2])
df_q8 = clean(df_q8, 'TRPTRANS', list(mode_labels.keys()))
df_q8['medcond_label'] = df_q8['MEDCOND'].map(
    {1: 'Has Medical Condition', 2: 'No Medical Condition'})
df_q8['mode_label'] = df_q8['TRPTRANS'].map(mode_labels)
ct8 = df_q8.groupby(['medcond_label','mode_label']).size().unstack(fill_value=0)
ct8_pct = ct8.div(ct8.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(12, 6))
ct8_pct.T.plot(kind='bar', ax=ax, color=[PALETTE[3], PALETTE[4]],
               edgecolor='white', width=0.7)
ax.set_ylabel('Share of Trips (%)')
ax.set_xlabel('Trip Mode')
ax.set_title('Q8. Trip Mode Share: With vs Without Medical Condition',
             fontweight='bold', fontsize=13)
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Health Status', fontsize=10)
save('q08_mode_by_medcond.png')
# Q9. Commute time distribution by occupation category
print("Q9: Commute time distribution by occupation category...")
occat_labels = {
    1: 'Management /\nProfessional',
    2: 'Service',
    3: 'Sales /\nOffice',
    4: 'Nat.Res /\nConstruction',
    97: 'Other'
}
df_q9 = clean(per, 'OCCAT', list(occat_labels.keys()))
df_q9 = df_q9[df_q9['TIMETOWK'] >= 0]
data_list = [df_q9[df_q9['OCCAT'] == k]['TIMETOWK'].values
             for k in occat_labels.keys()]
labels9 = list(occat_labels.values())
fig, ax = plt.subplots(figsize=(10, 6))
bp9 = ax.boxplot(data_list, labels=labels9, patch_artist=True,
                 showfliers=False, widths=0.5)
for patch, color in zip(bp9['boxes'], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Commute Time (minutes)')
ax.set_xlabel('Occupation Category')
ax.set_title('Q9. Commute Time Distribution by Occupation Category',
             fontweight='bold', fontsize=13)
save('q09_commute_by_occupation.png')
# Q10. Daily trip count: work-from-home vs commuters
print("Q10: Daily trip count — work-from-home vs commuters...")
wrk_labels = {1: 'Works from Home', 2: 'Commutes to Work'}
df_q10 = clean(merged, 'WRK_HOME', [1, 2])
df_q10 = df_q10[df_q10['CNTTDHH'] >= 0]
df_q10['wrk_label'] = df_q10['WRK_HOME'].map(wrk_labels)
mean_vals = df_q10.groupby('wrk_label')['CNTTDHH'].mean()
std_vals  = df_q10.groupby('wrk_label')['CNTTDHH'].std()

fig, ax = plt.subplots(figsize=(7, 6))
bars10 = ax.bar(mean_vals.index, mean_vals.values,
                yerr=std_vals.values / np.sqrt(len(df_q10)),
                color=[PALETTE[0], PALETTE[1]], edgecolor='white',
                capsize=6, width=0.4)
ax.set_ylabel('Average Daily Trips per Household')
ax.set_xlabel('Work Arrangement')
ax.set_title('Q10. Daily Trip Count:\nWork-from-Home vs Commuters',
             fontweight='bold', fontsize=13)
for bar, val in zip(bars10, mean_vals.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
            f'{val:.2f}', ha='center', va='bottom', fontsize=11)
save('q10_trips_wfh_vs_commute.png')
print("\nAll done! 10 individual plots have been saved.")
