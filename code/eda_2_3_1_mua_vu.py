"""
2.3.1. Những biểu hiện tính mùa vụ
Output: output/charts/2_3_1_mua_vu.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
import os

os.makedirs('output/charts', exist_ok=True)
sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'savefig.facecolor': 'white', 'font.size': 10})

print("Đang đọc dữ liệu...")
df = pd.read_csv('processed_energy_feature_data.csv', index_col=0, parse_dates=True)
region_cols = [c for c in ['AEP','COMED','DAYTON','DEOK','DOM','DUQ','EKPC','FE','NI','PJME','PJMW'] if c in df.columns]
colors = sns.color_palette('tab10', len(region_cols))
month_labels = ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12']
season_map   = {12:1,1:1,2:1, 3:2,4:2,5:2, 6:3,7:3,8:3, 9:4,10:4,11:4}
season_name  = {1:'Đông (12-2)',2:'Xuân (3-5)',3:'Hè (6-8)',4:'Thu (9-11)'}
season_color = {1:'royalblue',2:'mediumseagreen',3:'tomato',4:'goldenrod'}
df['season'] = df['month'].map(season_map)

fig = plt.figure(figsize=(20, 16))
fig.suptitle('2.3.1 — Tính Mùa Vụ Trong Tiêu Thụ Điện PJM', fontsize=18, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

# --- 1. Xu hướng dài hạn ---
ax1 = fig.add_subplot(gs[0, :])
monthly = df['AEP'].resample('ME').mean()
annual  = df['AEP'].resample('YE').mean()
ax1.plot(monthly.index, monthly.values, color='steelblue', linewidth=1, alpha=0.5, label='TB tháng')
ax1.plot(annual.index,  annual.values,  color='crimson',   linewidth=2.5, marker='o', markersize=7, label='TB năm')
for year in range(2005, 2019):
    ax1.axvspan(pd.Timestamp(f'{year}-06-01'), pd.Timestamp(f'{year}-08-31'),
                alpha=0.08, color='orange', label='Mùa hè' if year==2005 else '')
    end = f'{year+1}-02-28' if year < 2018 else '2018-07-01'
    ax1.axvspan(pd.Timestamp(f'{year}-12-01'), pd.Timestamp(end),
                alpha=0.08, color='lightblue', label='Mùa đông' if year==2005 else '')
ax1.set_title('Xu Hướng Dài Hạn Tải Điện AEP (2005–2018)', fontweight='bold')
ax1.set_ylabel('MW')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
ax1.legend(ncol=4, fontsize=9)

# --- 2. Mùa vụ theo tháng (chuẩn hoá) ---
ax2 = fig.add_subplot(gs[1, 0])
for i, col in enumerate(region_cols):
    m = df.groupby('month')[col].mean()
    m_norm = (m - m.min()) / (m.max() - m.min()) * 100
    ax2.plot(m.index, m_norm.values, color=colors[i], linewidth=1.8, marker='o', markersize=4, label=col)
ax2.set_xticks(range(1,13)); ax2.set_xticklabels(month_labels)
ax2.set_title('Mùa Vụ Theo Tháng (chuẩn hoá 0–100%)', fontweight='bold')
ax2.set_ylabel('Mức tải chuẩn hoá (%)')
ax2.legend(ncol=2, fontsize=7, loc='lower center')

# --- 3. Mùa vụ theo giờ — 4 mùa ---
ax3 = fig.add_subplot(gs[1, 1])
for s, name in season_name.items():
    h = df[df['season']==s].groupby('hour')['AEP'].mean()
    ax3.plot(h.index, h.values, color=season_color[s], linewidth=2.5, marker='o', markersize=4, label=name)
ax3.set_title('Mùa Vụ Theo Giờ — Phân Tách 4 Mùa (AEP)', fontweight='bold')
ax3.set_xlabel('Giờ'); ax3.set_ylabel('MW trung bình')
ax3.set_xticks(range(0,24,2))
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
ax3.legend(fontsize=9)

# --- 4. Heatmap giờ x tháng ---
ax4 = fig.add_subplot(gs[2, 0])
pivot = df.pivot_table(values='AEP', index='hour', columns='month', aggfunc='mean')
pivot.columns = month_labels
sns.heatmap(pivot, cmap='RdYlBu_r', ax=ax4, cbar_kws={'label':'MW'},
            linewidths=0.3, linecolor='white')
ax4.set_title('Heatmap Giờ × Tháng (AEP)\nGiao thoa mùa vụ ngày & năm', fontweight='bold')
ax4.set_xlabel('Tháng'); ax4.set_ylabel('Giờ')

# --- 5. Boxplot theo quý ---
ax5 = fig.add_subplot(gs[2, 1])
quarter_data = [df[df['quarter']==q]['AEP'].dropna().values for q in [1,2,3,4]]
bp = ax5.boxplot(quarter_data,
                 labels=['Q1\n(Đông/Xuân)','Q2\n(Xuân/Hè)','Q3\n(Hè/Thu)','Q4\n(Thu/Đông)'],
                 patch_artist=True, medianprops=dict(color='black', linewidth=2))
for patch, c in zip(bp['boxes'], ['royalblue','mediumseagreen','tomato','goldenrod']):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax5.set_title('Phân Phối Tải Điện Theo Quý (AEP)', fontweight='bold')
ax5.set_ylabel('MW')
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))

plt.savefig('output/charts/2_3_1_mua_vu.png', bbox_inches='tight')
plt.close()
print("✅ Đã lưu: output/charts/2_3_1_mua_vu.png")