"""
RAW_03. Khoảng Thời Gian & Độ Phủ Từng Vùng
Output: output/charts/raw_03_time_coverage.png
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
# Đọc từng file riêng để lấy khoảng thời gian chính xác
import glob
data_files = glob.glob('data/*_hourly.csv')
dfs_info = []
dfs_dict = {}
for f in sorted(data_files):
    region = os.path.basename(f).replace('_hourly.csv', '')
    try:
        tmp = pd.read_csv(f, parse_dates=['Datetime'], index_col='Datetime')
        tmp = tmp.sort_index()
        tmp = tmp[~tmp.index.duplicated(keep='first')]
        col = tmp.columns[0]
        dfs_info.append({
            'region': region,
            'start': tmp.index.min(),
            'end': tmp.index.max(),
            'n_total': len(tmp),
            'n_valid': tmp[col].notna().sum(),
            'missing_pct': tmp[col].isna().mean() * 100,
            'mean_mw': tmp[col].mean(),
            'std_mw': tmp[col].std(),
        })
        dfs_dict[region] = tmp[col]
    except Exception as e:
        print(f"  Bỏ qua {f}: {e}")

# Cũng đọc pjm_hourly_est.csv
df_merged = pd.read_csv('data/pjm_hourly_est.csv')
df_merged['Datetime'] = pd.to_datetime(df_merged['Datetime'])
df_merged = df_merged.set_index('Datetime').sort_index()
df_merged = df_merged[~df_merged.index.duplicated(keep='first')]

info = pd.DataFrame(dfs_info).set_index('region')
colors = sns.color_palette('tab20', len(info))

fig = plt.figure(figsize=(20, 20))
fig.suptitle('RAW 03 — Khoảng Thời Gian & Độ Phủ Dữ Liệu Từng Vùng', fontsize=18, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32)

# --- 1. Gantt chart: khoảng thời gian có dữ liệu ---
ax1 = fig.add_subplot(gs[0, :])
regions_sorted = info.sort_values('start').index.tolist()
for i, region in enumerate(regions_sorted):
    row = info.loc[region]
    duration = (row['end'] - row['start']).days / 365.25
    ax1.barh(i, duration, left=row['start'].year + row['start'].dayofyear/365,
             height=0.6, color=colors[i], alpha=0.85, edgecolor='white')
    ax1.text(row['start'].year + row['start'].dayofyear/365 + duration + 0.05,
             i, f"{row['start'].strftime('%Y')}→{row['end'].strftime('%Y')}",
             va='center', fontsize=8)

ax1.set_yticks(range(len(regions_sorted)))
ax1.set_yticklabels(regions_sorted, fontsize=10)
ax1.set_title('Khoảng Thời Gian Có Dữ Liệu Từng Vùng (Gantt Chart)', fontweight='bold')
ax1.set_xlabel('Năm')
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

# --- 2. Bar: số giờ hợp lệ từng vùng ---
ax2 = fig.add_subplot(gs[1, 0])
valid_sorted = info['n_valid'].sort_values(ascending=True)
bar_colors2 = [colors[regions_sorted.index(r)] if r in regions_sorted else 'steelblue' for r in valid_sorted.index]
bars2 = ax2.barh(valid_sorted.index, valid_sorted.values, color=bar_colors2, alpha=0.85, edgecolor='white')
ax2.set_title('Số Giờ Dữ Liệu Hợp Lệ Từng Vùng', fontweight='bold')
ax2.set_xlabel('Số giờ hợp lệ')
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
for bar, val in zip(bars2, valid_sorted.values):
    ax2.text(val + 500, bar.get_y() + bar.get_height()/2,
             f'{val:,}', va='center', fontsize=8)

# --- 3. Scatter: thời lượng dữ liệu vs missing% ---
ax3 = fig.add_subplot(gs[1, 1])
for i, region in enumerate(info.index):
    row = info.loc[region]
    duration_yr = (row['end'] - row['start']).days / 365.25
    ax3.scatter(duration_yr, row['missing_pct'], s=120,
                color=colors[i % len(colors)], alpha=0.85, zorder=3)
    ax3.annotate(region, (duration_yr, row['missing_pct']),
                 textcoords='offset points', xytext=(5, 3), fontsize=8)
ax3.set_title('Thời Lượng vs Tỷ Lệ Missing\n(mỗi điểm = 1 vùng)', fontweight='bold')
ax3.set_xlabel('Thời lượng có dữ liệu (năm)')
ax3.set_ylabel('% dữ liệu thiếu')
ax3.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Ngưỡng 50%')
ax3.legend(fontsize=9)

# --- 4. Chuỗi thời gian tổng hợp — số vùng có dữ liệu theo tháng ---
ax4 = fig.add_subplot(gs[2, :])
monthly_coverage = df_merged.notna().resample('ME').sum().gt(0).sum(axis=1).astype(float)
ax4.fill_between(monthly_coverage.index, monthly_coverage.values,
                 alpha=0.4, color='steelblue')
ax4.plot(monthly_coverage.index, monthly_coverage.values,
         color='steelblue', linewidth=1.5)
ax4.set_title('Số Vùng Có Dữ Liệu Hợp Lệ Theo Tháng (pjm_hourly_est.csv)', fontweight='bold')
ax4.set_xlabel('Thời gian')
ax4.set_ylabel('Số vùng có dữ liệu')
ax4.set_ylim(0, df_merged.shape[1] + 1)
ax4.set_yticks(range(0, df_merged.shape[1] + 1, 2))

# Highlight giai đoạn đủ dữ liệu
ax4.axhline(df_merged.shape[1] * 0.5, color='orange', linestyle='--',
            linewidth=1.5, label=f'50% vùng ({df_merged.shape[1]//2} vùng)')
ax4.axhline(df_merged.shape[1] * 0.8, color='green', linestyle='--',
            linewidth=1.5, label=f'80% vùng ({int(df_merged.shape[1]*0.8)} vùng)')
ax4.legend(fontsize=10)

plt.savefig('output/charts/raw_03_time_coverage.png', bbox_inches='tight')
plt.close()
print("✅ Đã lưu: output/charts/raw_03_time_coverage.png")