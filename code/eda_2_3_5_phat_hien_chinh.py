"""
2.3.5. Những phát hiện chính — Infographic tổng hợp
Output: output/charts/2_3_5_phat_hien_chinh.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import os

os.makedirs('output/charts', exist_ok=True)
plt.rcParams.update({'figure.dpi': 150, 'savefig.facecolor': '#f8f9fa', 'font.size': 10})

print("Đang đọc dữ liệu...")
df = pd.read_csv('processed_energy_feature_data.csv', index_col=0, parse_dates=True)
region_cols = [c for c in ['AEP','COMED','DAYTON','DEOK','DOM','DUQ','EKPC','FE','NI','PJME','PJMW'] if c in df.columns]
colors_tab = sns.color_palette('tab10', len(region_cols))
season_map  = {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}
df['season'] = df['month'].map(season_map)

# ==== Tính các số liệu chính ====
means        = df[region_cols].mean()
top_region   = means.idxmax(); top_val = means.max()
bot_region   = means.idxmin(); bot_val = means.min()
corr_mat     = df[region_cols].corr().copy()
corr_arr     = corr_mat.values.copy()
np.fill_diagonal(corr_arr, np.nan)
corr_mat_nodiag = pd.DataFrame(corr_arr, index=corr_mat.index, columns=corr_mat.columns)
max_corr_pair = corr_mat_nodiag.stack().idxmax()
min_corr_pair = corr_mat_nodiag.stack().idxmin()
peak_hour     = df.groupby('hour')['AEP'].mean().idxmax()
trough_hour   = df.groupby('hour')['AEP'].mean().idxmin()
weekday_mean  = df[df['is_weekend']==0]['AEP'].mean()
weekend_mean  = df[df['is_weekend']==1]['AEP'].mean()
pct_drop      = (weekday_mean - weekend_mean) / weekday_mean * 100
summer_mean   = df[df['season']==3]['AEP'].mean()
winter_mean   = df[df['season']==1]['AEP'].mean()
spring_mean   = df[df['season']==2]['AEP'].mean()

Q1c, Q3c = df['AEP'].quantile(0.25), df['AEP'].quantile(0.75)
IQR = Q3c - Q3c
outlier_pct = ((df['AEP'] < Q1c-1.5*(Q3c-Q1c)) | (df['AEP'] > Q3c+1.5*(Q3c-Q1c))).sum() / len(df) * 100
data_quality_mean = df['data_quality'].mean() if 'data_quality' in df.columns else 0.7

fig = plt.figure(figsize=(20, 20), facecolor='#f8f9fa')
fig.suptitle('2.3.5 — Những Phát Hiện Chính Từ Phân Tích Dữ Liệu PJM',
             fontsize=20, fontweight='bold', y=0.99, color='#1a1a2e')
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

# ===== PANEL 1: KPI cards (dạng text summary) =====
ax_kpi = fig.add_subplot(gs[0, :])
ax_kpi.set_xlim(0, 10); ax_kpi.set_ylim(0, 3)
ax_kpi.axis('off')
ax_kpi.set_facecolor('#f8f9fa')

kpis = [
    ('🏆 Vùng lớn nhất', f'{top_region}\n{top_val:,.0f} MW', '#2563eb'),
    ('📉 Vùng nhỏ nhất', f'{bot_region}\n{bot_val:,.0f} MW', '#16a34a'),
    ('⚡ Giờ cao điểm', f'{peak_hour}:00\n(đỉnh tiêu thụ)', '#dc2626'),
    ('🌙 Giờ thấp điểm', f'{trough_hour}:00\n(đáy tiêu thụ)', '#9333ea'),
    ('📅 Cuối tuần -', f'{pct_drop:.1f}%\nso với ngày thường', '#ea580c'),
]
for i, (title, value, color) in enumerate(kpis):
    x = 1 + i * 2
    rect = mpatches.FancyBboxPatch((x-0.85, 0.2), 1.7, 2.5,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, alpha=0.12,
                                    edgecolor=color, linewidth=2)
    ax_kpi.add_patch(rect)
    ax_kpi.text(x, 2.3, title, ha='center', va='center', fontsize=11,
                fontweight='bold', color=color)
    ax_kpi.text(x, 1.3, value, ha='center', va='center', fontsize=13,
                fontweight='bold', color='#1a1a2e')

ax_kpi.set_title('Tóm Tắt Các Chỉ Số Quan Trọng', fontweight='bold', fontsize=14, pad=10)

# ===== PANEL 2: So sánh mùa (bar) =====
ax2 = fig.add_subplot(gs[1, 0])
season_means = df.groupby('season')['AEP'].mean()
season_labels = ['Đông\n(12-2)', 'Xuân\n(3-5)', 'Hè\n(6-8)', 'Thu\n(9-11)']
season_colors = ['royalblue', 'mediumseagreen', 'tomato', 'goldenrod']
bars = ax2.bar(season_labels, [season_means.get(s, 0) for s in [1,2,3,4]],
               color=season_colors, alpha=0.85, edgecolor='white', linewidth=1.5)
ax2.set_title('Phát Hiện 1: Mùa Hè & Mùa Đông\ncó Tải Điện Cao Nhất', fontweight='bold')
ax2.set_ylabel('MW trung bình')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
for bar, s in zip(bars, [1,2,3,4]):
    val = season_means.get(s, 0)
    ax2.text(bar.get_x()+bar.get_width()/2, val+30,
             f'{val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
best_s = season_means.idxmax()
ax2.text(0.98, 0.97, f'Mùa cao nhất: {season_labels[[1,2,3,4].index(best_s)].replace(chr(10)," ")}',
         transform=ax2.transAxes, ha='right', va='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# ===== PANEL 3: Đường cong giờ ngày thường vs cuối tuần =====
ax3 = fig.add_subplot(gs[1, 1])
for label, mask, color, lw in [
    ('Ngày thường', (df['is_weekend']==0)&(df['is_holiday']==0), 'steelblue', 2.5),
    ('Cuối tuần',   df['is_weekend']==1,                          'coral',     2.5),
    ('Ngày lễ',     df['is_holiday']==1,                          'green',     2),
]:
    h = df[mask].groupby('hour')['AEP'].mean()
    ax3.plot(h.index, h.values, color=color, linewidth=lw, marker='o', markersize=3, label=label)
ax3.fill_between(range(24),
                 df[df['is_weekend']==0].groupby('hour')['AEP'].mean(),
                 df[df['is_weekend']==1].groupby('hour')['AEP'].mean(),
                 alpha=0.15, color='gray', label='Khoảng chênh lệch')
ax3.set_title('Phát Hiện 2: Ngày Thường Tiêu Thụ\nNhiều Hơn Cuối Tuần Toàn Bộ 24 Giờ', fontweight='bold')
ax3.set_xlabel('Giờ'); ax3.set_ylabel('MW')
ax3.set_xticks(range(0,24,2))
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
ax3.legend(fontsize=9)

# ===== PANEL 4: Top tương quan =====
ax4 = fig.add_subplot(gs[2, 0])
corr_vals = corr_mat_nodiag.stack().dropna().sort_values(ascending=False)
top10 = corr_vals.head(10)
labels_10 = [f'{a}–{b}' for a,b in top10.index]
bar_colors_10 = ['#2563eb' if v > 0.9 else '#16a34a' if v > 0.8 else '#ea580c' for v in top10.values]
bars4 = ax4.barh(labels_10[::-1], top10.values[::-1], color=bar_colors_10[::-1], alpha=0.85)
ax4.set_xlim(0.5, 1.02)
ax4.set_title('Phát Hiện 3: Top 10 Cặp Vùng\nCó Tương Quan Cao Nhất', fontweight='bold')
ax4.set_xlabel('Hệ số tương quan Pearson')
for bar, val in zip(bars4, top10.values[::-1]):
    ax4.text(val + 0.002, bar.get_y()+bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=9)

# ===== PANEL 5: Tổng hợp chất lượng + xu hướng =====
ax5 = fig.add_subplot(gs[2, 1])
yearly = df.groupby('year')['AEP'].mean()
ax5.bar(yearly.index, yearly.values, color='steelblue', alpha=0.6, label='TB năm')
z = np.polyfit(yearly.index, yearly.values, 1)
p = np.poly1d(z)
ax5.plot(yearly.index, p(yearly.index), color='crimson', linewidth=2.5,
         linestyle='--', label=f'Xu hướng: {z[0]:+.0f} MW/năm')
ax5.set_title('Phát Hiện 4: Xu Hướng Dài Hạn\nTải Điện AEP Theo Năm', fontweight='bold')
ax5.set_xlabel('Năm'); ax5.set_ylabel('MW trung bình')
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
ax5.legend(fontsize=10)
direction = 'giảm' if z[0] < 0 else 'tăng'
ax5.text(0.98, 0.05, f'Xu hướng {direction} ~{abs(z[0]):.0f} MW/năm\n(~{abs(z[0])/yearly.mean()*100:.2f}%/năm)',
         transform=ax5.transAxes, ha='right', va='bottom', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.savefig('output/charts/2_3_5_phat_hien_chinh.png', bbox_inches='tight')
plt.close()
print("✅ Đã lưu: output/charts/2_3_5_phat_hien_chinh.png")
