"""
RAW_04. Tương Quan Giữa Các Vùng (Dữ Liệu Gốc)
Output: output/charts/raw_04_tuong_quan_raw.png
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
fig.suptitle('RAW 04 — Tương Quan Giữa Các Vùng (Dữ Liệu Gốc)', fontsize=18, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

# --- 1. Heatmap tương quan đầy đủ ---
ax1 = fig.add_subplot(gs[0, :])
corr = df[region_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # Hiện cả ma trận
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            vmin=0.4, vmax=1.0, ax=ax1,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Pearson r', 'shrink': 0.6},
            annot_kws={'size': 9})
ax1.set_title('Ma Trận Tương Quan Pearson (Dữ Liệu Gốc — Có NaN)', fontweight='bold')
ax1.tick_params(axis='x', rotation=30)
ax1.tick_params(axis='y', rotation=0)

# --- 2. Bar: top & bottom tương quan (loại đường chéo) ---
ax2 = fig.add_subplot(gs[1, 0])
corr_arr = corr.values.copy()
np.fill_diagonal(corr_arr, np.nan)
corr_nodiag = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)
corr_vals = corr_nodiag.stack().dropna().sort_values(ascending=False)
# Lọc bỏ cặp trùng (A-B và B-A)
seen = set()
unique_pairs = []
for (a, b), v in corr_vals.items():
    key = frozenset([a, b])
    if key not in seen:
        seen.add(key)
        unique_pairs.append((f'{a}–{b}', v))
unique_pairs_df = pd.DataFrame(unique_pairs, columns=['pair', 'corr'])

top8 = unique_pairs_df.head(8)
bar_colors_top = ['#2ca02c' if v > 0.85 else '#ff7f0e' for v in top8['corr']]
bars_top = ax2.barh(top8['pair'][::-1], top8['corr'][::-1], color=bar_colors_top[::-1], alpha=0.85)
ax2.set_xlim(0.5, 1.02)
ax2.set_title('Top 8 Cặp Vùng Tương Quan Cao Nhất', fontweight='bold')
ax2.set_xlabel('Hệ số tương quan')
for bar, val in zip(bars_top, top8['corr'][::-1]):
    ax2.text(val + 0.003, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=9)

# --- 3. Bottom tương quan ---
ax3 = fig.add_subplot(gs[1, 1])
bot8 = unique_pairs_df[unique_pairs_df['corr'].notna()].tail(8).sort_values('corr')
bar_colors_bot = ['#d62728' if v < 0.6 else '#ff7f0e' for v in bot8['corr']]
bars_bot = ax3.barh(bot8['pair'], bot8['corr'], color=bar_colors_bot, alpha=0.85)
ax3.set_xlim(0, 1.02)
ax3.set_title('8 Cặp Vùng Tương Quan Thấp Nhất\n(có thể do ít dữ liệu chung)', fontweight='bold')
ax3.set_xlabel('Hệ số tương quan')
for bar, val in zip(bars_bot, bot8['corr']):
    ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}' if not np.isnan(val) else 'NaN', va='center', fontsize=9)

# --- 4. Scatter matrix 4 vùng chính ---
main_4 = [c for c in ['AEP', 'PJME', 'DOM', 'COMED'] if c in region_cols]
gs_inner = gridspec.GridSpecFromSubplotSpec(len(main_4), len(main_4),
                                             subplot_spec=gs[2, :],
                                             hspace=0.15, wspace=0.15)
sample = df[main_4].dropna().sample(min(3000, len(df[main_4].dropna())), random_state=42)
for i, col_y in enumerate(main_4):
    for j, col_x in enumerate(main_4):
        ax_s = fig.add_subplot(gs_inner[i, j])
        if i == j:
            ax_s.hist(df[col_x].dropna(), bins=40, color=colors[j], alpha=0.7, edgecolor='white')
            ax_s.set_ylabel(col_y if j == 0 else '')
        else:
            ax_s.scatter(sample[col_x], sample[col_y], alpha=0.15, s=5,
                         color=colors[j])
            r = sample[[col_x, col_y]].corr().iloc[0, 1]
            ax_s.text(0.05, 0.92, f'r={r:.2f}', transform=ax_s.transAxes,
                      fontsize=8, color='crimson', fontweight='bold')
        if i == len(main_4)-1:
            ax_s.set_xlabel(col_x, fontsize=9)
        if j == 0:
            ax_s.set_ylabel(col_y, fontsize=9)
        ax_s.tick_params(labelsize=7)
        ax_s.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
        ax_s.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))

fig.text(0.5, 0.02, 'Scatter Matrix: AEP, PJME, DOM, COMED (mẫu 3,000 điểm)',
         ha='center', fontsize=12, fontweight='bold')

plt.savefig('output/charts/raw_04_tuong_quan_raw.png', bbox_inches='tight')
plt.close()
print("✅ Đã lưu: output/charts/raw_04_tuong_quan_raw.png")