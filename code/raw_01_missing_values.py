"""
RAW_01. Missing Values & Chất Lượng Dữ Liệu Gốc
Output: output/charts/raw_01_missing_values.png
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

print("Đang đọc dữ liệu gốc...")
df = pd.read_csv('data/pjm_hourly_est.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime').sort_index()
df = df[~df.index.duplicated(keep='first')]

region_cols = df.columns.tolist()
colors = sns.color_palette('tab10', len(region_cols))

print(f"Shape gốc: {df.shape}")
print(f"Thời gian: {df.index.min()} → {df.index.max()}")

fig = plt.figure(figsize=(20, 18))
fig.suptitle('RAW 01 — Missing Values & Chất Lượng Dữ Liệu Gốc PJM', fontsize=18, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

# --- 1. Tỷ lệ missing % từng vùng ---
ax1 = fig.add_subplot(gs[0, 0])
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
bar_colors = ['#d62728' if v > 70 else '#ff7f0e' if v > 40 else '#2ca02c' for v in missing_pct.values]
bars = ax1.barh(missing_pct.index, missing_pct.values, color=bar_colors, alpha=0.85, edgecolor='white')
ax1.axvline(70, color='#d62728', linestyle='--', linewidth=1.5, label='Ngưỡng 70%')
ax1.axvline(40, color='#ff7f0e', linestyle='--', linewidth=1.5, label='Ngưỡng 40%')
ax1.set_title('Tỷ Lệ Dữ Liệu Thiếu Theo Vùng (%)', fontweight='bold')
ax1.set_xlabel('% dữ liệu thiếu')
for bar, val in zip(bars, missing_pct.values):
    ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9)
ax1.legend(fontsize=9)

# --- 2. Heatmap missing theo năm x vùng ---
ax2 = fig.add_subplot(gs[0, 1])
df_year = df.copy()
df_year['year'] = df_year.index.year
missing_by_year = df_year.groupby('year')[region_cols].apply(lambda x: x.isnull().mean() * 100)
sns.heatmap(missing_by_year.T, cmap='Reds', ax=ax2, cbar_kws={'label': '% missing'},
            linewidths=0.3, linecolor='white', annot=True, fmt='.0f', annot_kws={'size': 8})
ax2.set_title('Tỷ Lệ Missing Theo Năm × Vùng (%)', fontweight='bold')
ax2.set_xlabel('Năm'); ax2.set_ylabel('Vùng')

# --- 3. Timeline missing — mỗi vùng có dữ liệu năm nào ---
ax3 = fig.add_subplot(gs[1, :])
years = sorted(df.index.year.unique())
for i, col in enumerate(region_cols):
    for year in years:
        year_data = df[df.index.year == year][col]
        pct_valid = year_data.notna().mean()
        color = '#2ca02c' if pct_valid > 0.9 else '#ff7f0e' if pct_valid > 0.5 else '#d62728' if pct_valid > 0 else '#cccccc'
        ax3.barh(i, 1, left=year - 0.5, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        if pct_valid > 0:
            ax3.text(year, i, f'{pct_valid*100:.0f}', ha='center', va='center', fontsize=6, color='white', fontweight='bold')

ax3.set_yticks(range(len(region_cols)))
ax3.set_yticklabels(region_cols)
ax3.set_xticks(years)
ax3.set_xticklabels([str(y) for y in years], rotation=45)
ax3.set_title('Độ Phủ Dữ Liệu Theo Năm & Vùng (% dữ liệu hợp lệ)', fontweight='bold')
ax3.set_xlabel('Năm')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ca02c', label='>90% hợp lệ'),
    Patch(facecolor='#ff7f0e', label='50–90% hợp lệ'),
    Patch(facecolor='#d62728', label='<50% hợp lệ'),
    Patch(facecolor='#cccccc', label='Không có dữ liệu'),
]
ax3.legend(handles=legend_elements, loc='lower right', fontsize=9, ncol=4)

# --- 4. Phân phối số vùng có dữ liệu mỗi giờ ---
ax4 = fig.add_subplot(gs[2, 0])
valid_per_row = df.notna().sum(axis=1)
ax4.hist(valid_per_row.values, bins=len(region_cols)+1,
         color='steelblue', alpha=0.8, edgecolor='white', linewidth=0.5,
         range=(-0.5, len(region_cols)+0.5))
ax4.set_title('Số Vùng Có Dữ Liệu Hợp Lệ Mỗi Giờ', fontweight='bold')
ax4.set_xlabel('Số vùng có dữ liệu')
ax4.set_ylabel('Số giờ')
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax4.set_xticks(range(len(region_cols)+1))
total = len(df)
for x in range(len(region_cols)+1):
    cnt = (valid_per_row == x).sum()
    if cnt > 0:
        ax4.text(x, cnt + total*0.002, f'{cnt/total*100:.1f}%',
                 ha='center', va='bottom', fontsize=7, rotation=90)

# --- 5. Missing theo tháng (tổng hợp) ---
ax5 = fig.add_subplot(gs[2, 1])
df['month_'] = df.index.month
missing_by_month = df.groupby('month_')[region_cols].apply(lambda x: x.isnull().mean() * 100).mean(axis=1)
month_labels = ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12']
bar_colors_m = ['#d62728' if v > 70 else '#ff7f0e' if v > 40 else '#2ca02c' for v in missing_by_month.values]
ax5.bar(month_labels, missing_by_month.values, color=bar_colors_m, alpha=0.85, edgecolor='white')
ax5.set_title('Tỷ Lệ Missing Trung Bình Theo Tháng\n(trung bình tất cả vùng)', fontweight='bold')
ax5.set_xlabel('Tháng'); ax5.set_ylabel('% missing trung bình')
for i, val in enumerate(missing_by_month.values):
    ax5.text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

plt.savefig('output/charts/raw_01_missing_values.png', bbox_inches='tight')
plt.close()
print("✅ Đã lưu: output/charts/raw_01_missing_values.png")