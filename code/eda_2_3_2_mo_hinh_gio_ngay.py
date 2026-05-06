"""
2.3.2. Mô hình theo giờ và theo ngày
Output: output/charts/2_3_2_mo_hinh_gio_ngay.png
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
day_labels = ['T2','T3','T4','T5','T6','T7','CN']
season_map  = {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}
season_name = {1:'Đông',2:'Xuân',3:'Hè',4:'Thu'}
season_color= {1:'royalblue',2:'mediumseagreen',3:'tomato',4:'goldenrod'}
df['season'] = df['month'].map(season_map)

fig = plt.figure(figsize=(20, 18))
fig.suptitle('2.3.2 — Mô Hình Tiêu Thụ Theo Giờ & Theo Ngày', fontsize=18, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

# --- 1. Đường cong giờ trung bình — tất cả vùng ---
ax1 = fig.add_subplot(gs[0, 0])
for i, col in enumerate(region_cols):
    h = df.groupby('hour')[col].mean()
    h_norm = (h - h.min()) / (h.max() - h.min()) * 100
    ax1.plot(h.index, h_norm.values, color=colors[i], linewidth=1.8, label=col)
ax1.set_title('Đường Cong Tiêu Thụ Theo Giờ\n(chuẩn hoá — tất cả vùng)', fontweight='bold')
ax1.set_xlabel('Giờ'); ax1.set_ylabel('Mức tải chuẩn hoá (%)')
ax1.set_xticks(range(0,24,2))
ax1.legend(ncol=2, fontsize=7)
ax1.axvspan(7, 10, alpha=0.1, color='red', label='Đỉnh sáng')
ax1.axvspan(17, 21, alpha=0.1, color='orange', label='Đỉnh chiều')

# --- 2. Đường cong giờ AEP — phân tách ngày thường vs cuối tuần ---
ax2 = fig.add_subplot(gs[0, 1])
for label, mask, color, ls in [
    ('Ngày thường', (df['is_weekend']==0)&(df['is_holiday']==0), 'steelblue', '-'),
    ('Cuối tuần',   df['is_weekend']==1,                          'coral',     '--'),
    ('Ngày lễ',     df['is_holiday']==1,                          'green',     ':'),
]:
    h = df[mask].groupby('hour')['AEP'].mean()
    std = df[mask].groupby('hour')['AEP'].std()
    ax2.plot(h.index, h.values, color=color, linewidth=2.5, linestyle=ls, label=label)
    ax2.fill_between(h.index, h-std, h+std, alpha=0.12, color=color)
ax2.set_title('Đường Cong Giờ AEP\nNgày thường / Cuối tuần / Ngày lễ', fontweight='bold')
ax2.set_xlabel('Giờ'); ax2.set_ylabel('MW')
ax2.set_xticks(range(0,24,2))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
ax2.legend(fontsize=10)

# --- 3. Đường cong giờ theo 4 mùa (AEP) ---
ax3 = fig.add_subplot(gs[1, 0])
for s, name in season_name.items():
    h = df[df['season']==s].groupby('hour')['AEP'].mean()
    std = df[df['season']==s].groupby('hour')['AEP'].std()
    ax3.plot(h.index, h.values, color=season_color[s], linewidth=2.5, marker='o', markersize=3, label=name)
    ax3.fill_between(h.index, h-std, h+std, alpha=0.1, color=season_color[s])
ax3.set_title('Đường Cong Giờ Theo Mùa (AEP)\nvới dải ±1 std', fontweight='bold')
ax3.set_xlabel('Giờ'); ax3.set_ylabel('MW')
ax3.set_xticks(range(0,24,2))
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
ax3.legend(fontsize=10)

# --- 4. Mẫu theo ngày trong tuần — tất cả vùng ---
ax4 = fig.add_subplot(gs[1, 1])
for i, col in enumerate(region_cols):
    d = df.groupby('dayofweek')[col].mean()
    d_norm = (d - d.min()) / (d.max() - d.min()) * 100
    ax4.plot(d.index, d_norm.values, color=colors[i], linewidth=1.8, marker='s', markersize=5, label=col)
ax4.axvspan(4.5, 6.5, alpha=0.08, color='gray')
ax4.set_xticks(range(7)); ax4.set_xticklabels(day_labels)
ax4.set_title('Mẫu Tiêu Thụ Theo Ngày Trong Tuần\n(chuẩn hoá)', fontweight='bold')
ax4.set_ylabel('Mức tải chuẩn hoá (%)')
ax4.legend(ncol=2, fontsize=7)
ax4.text(5.5, 5, 'Cuối\ntuần', ha='center', fontsize=9, color='gray')

# --- 5. Boxplot giờ (AEP) — thấy rõ phân phối từng giờ ---
ax5 = fig.add_subplot(gs[2, :])
hourly_data = [df[df['hour']==h]['AEP'].dropna().values for h in range(24)]
bp = ax5.boxplot(hourly_data, labels=range(24), patch_artist=True,
                 medianprops=dict(color='black', linewidth=1.5),
                 flierprops=dict(marker='.', markersize=2, alpha=0.3))
cmap = plt.cm.RdYlBu_r
for h, patch in enumerate(bp['boxes']):
    # màu theo mức tải trung bình
    mean_val = df[df['hour']==h]['AEP'].mean()
    patch.set_facecolor(cmap((mean_val - 12000) / 6000))
    patch.set_alpha(0.8)
ax5.set_title('Phân Phối Tải Điện AEP Theo Từng Giờ Trong Ngày', fontweight='bold')
ax5.set_xlabel('Giờ'); ax5.set_ylabel('MW')
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))

plt.savefig('output/charts/2_3_2_mo_hinh_gio_ngay.png', bbox_inches='tight')
plt.close()
print("✅ Đã lưu: output/charts/2_3_2_mo_hinh_gio_ngay.png")