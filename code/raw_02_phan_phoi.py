"""
RAW_02. Phân Phối Tải Điện Từng Vùng (Dữ Liệu Gốc)
Output: output/charts/raw_02_phan_phoi.png
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

fig = plt.figure(figsize=(20, 20))
fig.suptitle('RAW 02 — Phân Phối Tải Điện Từng Vùng (Dữ Liệu Gốc)', fontsize=18, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.32)

# --- 1. Boxplot tất cả vùng ---
ax1 = fig.add_subplot(gs[0, :])
data_box = [df[c].dropna().values for c in region_cols]
bp = ax1.boxplot(data_box, labels=region_cols, patch_artist=True,
                 medianprops=dict(color='black', linewidth=2),
                 flierprops=dict(marker='.', markersize=3, alpha=0.3))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax1.set_title('Phân Phối Tải Điện Theo Vùng — Boxplot (Dữ Liệu Gốc)', fontweight='bold')
ax1.set_ylabel('MW')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax1.tick_params(axis='x', rotation=15)

# --- 2. Histogram từng vùng (3x4 grid) ---
n_cols_plot = 4
n_rows_plot = int(np.ceil(len(region_cols) / n_cols_plot))
gs_inner = gridspec.GridSpecFromSubplotSpec(n_rows_plot, n_cols_plot,
                                             subplot_spec=gs[1:3, :],
                                             hspace=0.6, wspace=0.35)
axes_hist = [fig.add_subplot(gs_inner[i // n_cols_plot, i % n_cols_plot]) for i in range(len(region_cols))]

for i, (col, ax) in enumerate(zip(region_cols, axes_hist)):
    data = df[col].dropna()
    ax.hist(data, bins=50, color=colors[i], alpha=0.75, edgecolor='white', linewidth=0.3)
    ax.axvline(data.mean(), color='black', linewidth=1.8, linestyle='--', label=f'Mean\n{data.mean():,.0f}')
    ax.axvline(data.median(), color='red', linewidth=1.5, linestyle=':', label=f'Median\n{data.median():,.0f}')

    # Skewness
    skew = data.skew()
    ax.set_title(f'{col}\nSkew={skew:.2f}', fontsize=10, fontweight='bold')
    ax.set_xlabel('MW', fontsize=8)
    ax.set_ylabel('Tần suất', fontsize=8)
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))

# --- 3. Thống kê tóm tắt dạng bảng ---
ax_table = fig.add_subplot(gs[3, :])
ax_table.axis('off')
stats = df[region_cols].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(0)
stats['count'] = stats['count'].astype(int)
stats['missing%'] = (df[region_cols].isnull().sum() / len(df) * 100).round(1)
stats = stats[['count', 'missing%', 'mean', 'std', 'min', '50%', 'max']]
stats.columns = ['N hợp lệ', 'Missing%', 'Mean', 'Std', 'Min', 'Median', 'Max']

table = ax_table.table(
    cellText=[[f'{v:,.0f}' if isinstance(v, float) and col not in ['Missing%'] else
               f'{v:.1f}%' if col == 'Missing%' else f'{v:,}' for col, v in zip(stats.columns, row)]
              for row in stats.values],
    rowLabels=stats.index,
    colLabels=stats.columns,
    cellLoc='center', loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(9)
for (r, c), cell in table.get_celld().items():
    if r == 0 or c == -1:
        cell.set_facecolor('#2563eb')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#f0f4ff')
ax_table.set_title('Bảng Thống Kê Mô Tả Từng Vùng (Dữ Liệu Gốc)', fontweight='bold', pad=10)

plt.savefig('output/charts/raw_02_phan_phoi.png', bbox_inches='tight')
plt.close()
print("✅ Đã lưu: output/charts/raw_02_phan_phoi.png")