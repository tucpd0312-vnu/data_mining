"""
2.3.3. So sánh giữa các vùng và mối tương quan
Output: output/charts/2_3_3_tuong_quan_vung.png
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

fig = plt.figure(figsize=(20, 18))
fig.suptitle('2.3.3 — So Sánh Giữa Các Vùng & Mối Tương Quan', fontsize=18, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

# --- 1. Bar chart tổng MW trung bình từng vùng ---
ax1 = fig.add_subplot(gs[0, 0])
means = df[region_cols].mean().sort_values(ascending=True)
bars = ax1.barh(means.index, means.values,
                color=[colors[region_cols.index(r)] for r in means.index], alpha=0.8)
ax1.set_title('Tải Điện Trung Bình Từng Vùng', fontweight='bold')
ax1.set_xlabel('MW trung bình')
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
for bar, val in zip(bars, means.values):
    ax1.text(val + 50, bar.get_y() + bar.get_height()/2,
             f'{val:,.0f}', va='center', fontsize=9)

# --- 2. Boxplot so sánh tất cả vùng ---
ax2 = fig.add_subplot(gs[0, 1])
data_box = [df[c].dropna().values for c in region_cols]
bp = ax2.boxplot(data_box, labels=region_cols, patch_artist=True,
                 medianprops=dict(color='black', linewidth=2),
                 flierprops=dict(marker='.', markersize=2, alpha=0.3))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax2.set_title('Phân Phối Tải Điện Theo Vùng (Boxplot)', fontweight='bold')
ax2.set_ylabel('MW')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
ax2.tick_params(axis='x', rotation=15)

# --- 3. Heatmap tương quan ---
ax3 = fig.add_subplot(gs[1, :])
corr = df[region_cols].corr().round(2)
mask = np.zeros_like(corr, dtype=bool)  # Hiện toàn bộ
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            vmin=0.5, vmax=1.0, ax=ax3,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Hệ số tương quan Pearson', 'shrink': 0.8},
            annot_kws={'size': 10})
ax3.set_title('Ma Trận Tương Quan Pearson Giữa Các Vùng', fontweight='bold')
ax3.tick_params(axis='x', rotation=0)
ax3.tick_params(axis='y', rotation=0)

# --- 4. Scatter: PJME vs AEP (2 vùng tương quan cao) ---
ax4 = fig.add_subplot(gs[2, 0])
sample = df[['PJME','AEP']].dropna().sample(5000, random_state=42)
ax4.scatter(sample['PJME'], sample['AEP'], alpha=0.15, s=8, color='steelblue')
# Đường hồi quy
z = np.polyfit(sample['PJME'], sample['AEP'], 1)
p = np.poly1d(z)
x_line = np.linspace(sample['PJME'].min(), sample['PJME'].max(), 100)
ax4.plot(x_line, p(x_line), color='crimson', linewidth=2, label=f'r = {sample.corr().loc["PJME","AEP"]:.2f}')
ax4.set_title('Scatter PJME vs AEP\n(mẫu 5,000 điểm)', fontweight='bold')
ax4.set_xlabel('PJME (MW)'); ax4.set_ylabel('AEP (MW)')
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
ax4.legend(fontsize=11)

# --- 5. Scatter: EKPC vs NI (2 vùng tương quan thấp hơn) ---
ax5 = fig.add_subplot(gs[2, 1])
sample2 = df[['EKPC','NI']].dropna().sample(5000, random_state=42)
ax5.scatter(sample2['EKPC'], sample2['NI'], alpha=0.15, s=8, color='darkorange')
z2 = np.polyfit(sample2['EKPC'], sample2['NI'], 1)
p2 = np.poly1d(z2)
x2 = np.linspace(sample2['EKPC'].min(), sample2['EKPC'].max(), 100)
ax5.plot(x2, p2(x2), color='crimson', linewidth=2, label=f'r = {sample2.corr().loc["EKPC","NI"]:.2f}')
ax5.set_title('Scatter EKPC vs NI\n(mẫu 5,000 điểm)', fontweight='bold')
ax5.set_xlabel('EKPC (MW)'); ax5.set_ylabel('NI (MW)')
ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
ax5.legend(fontsize=11)

plt.savefig('output/charts/2_3_3_tuong_quan_vung.png', bbox_inches='tight')
plt.close()
print("✅ Đã lưu: output/charts/2_3_3_tuong_quan_vung.png")