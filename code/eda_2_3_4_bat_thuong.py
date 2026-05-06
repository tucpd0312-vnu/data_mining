"""
2.3.4. Các bất thường và chất lượng dữ liệu
Output: output/charts/2_3_4_bat_thuong_chat_luong.png
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

# Đọc dữ liệu gốc để tính missing rate
df_raw = pd.read_csv('processed_data.csv', index_col=0, parse_dates=True)
region_cols = [c for c in ['AEP','COMED','DAYTON','DEOK','DOM','DUQ','EKPC','FE','NI','PJME','PJMW'] if c in df.columns]
colors = sns.color_palette('tab10', len(region_cols))

fig = plt.figure(figsize=(20, 18))
fig.suptitle('2.3.4 — Bất Thường & Chất Lượng Dữ Liệu', fontsize=18, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

# --- 1. Missing rate từng vùng (dữ liệu gốc) ---
ax1 = fig.add_subplot(gs[0, 0])
raw_region_cols = [c for c in region_cols if c in df_raw.columns]
missing_rate = (df_raw[raw_region_cols].isnull().sum() / len(df_raw) * 100).sort_values(ascending=True)
bar_colors = ['#d62728' if v > 50 else '#ff7f0e' if v > 30 else '#2ca02c' for v in missing_rate.values]
bars = ax1.barh(missing_rate.index, missing_rate.values, color=bar_colors, alpha=0.85)
ax1.axvline(50, color='red', linestyle='--', linewidth=1.5, label='Ngưỡng 50%')
ax1.axvline(30, color='orange', linestyle='--', linewidth=1.5, label='Ngưỡng 30%')
ax1.set_title('Tỷ Lệ Dữ Liệu Thiếu Theo Vùng\n(dữ liệu gốc trước xử lý)', fontweight='bold')
ax1.set_xlabel('% dữ liệu thiếu')
for bar, val in zip(bars, missing_rate.values):
    ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9)
ax1.legend(fontsize=9)

# --- 2. Phân phối data_quality ---
ax2 = fig.add_subplot(gs[0, 1])
if 'data_quality' in df.columns:
    dq = df['data_quality']
    counts = dq.value_counts().sort_index()
    ax2.bar(counts.index.astype(str), counts.values,
            color=['#d62728' if v < 0.6 else '#ff7f0e' if v < 0.8 else '#2ca02c' for v in counts.index],
            alpha=0.85, edgecolor='white')
    ax2.set_title('Phân Phối Điểm Chất Lượng Dữ Liệu\n(data_quality score)', fontweight='bold')
    ax2.set_xlabel('Điểm chất lượng (tỷ lệ vùng có dữ liệu thật)')
    ax2.set_ylabel('Số giờ')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
    for i, (x, y) in enumerate(zip(counts.index.astype(str), counts.values)):
        ax2.text(i, y + 100, f'{y:,}', ha='center', fontsize=9)
else:
    ax2.text(0.5, 0.5, 'Không có cột data_quality', ha='center', va='center', transform=ax2.transAxes)

# --- 3. Phát hiện outlier bằng IQR — đếm outlier từng vùng ---
ax3 = fig.add_subplot(gs[1, 0])
outlier_counts = {}
for col in region_cols:
    data = df[col].dropna()
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
    outlier_counts[col] = len(outliers)
outlier_s = pd.Series(outlier_counts).sort_values(ascending=True)
bars3 = ax3.barh(outlier_s.index, outlier_s.values,
                 color=[colors[region_cols.index(r)] for r in outlier_s.index], alpha=0.8)
ax3.set_title('Số Lượng Outlier Theo Vùng\n(phương pháp IQR: < Q1-1.5*IQR hoặc > Q3+1.5*IQR)', fontweight='bold')
ax3.set_xlabel('Số điểm outlier')
for bar, val in zip(bars3, outlier_s.values):
    ax3.text(val + 1, bar.get_y() + bar.get_height()/2,
             f'{val:,}', va='center', fontsize=9)

# --- 4. Time series AEP với outlier được đánh dấu ---
ax4 = fig.add_subplot(gs[1, 1])
data_aep = df['AEP'].dropna()
Q1, Q3 = data_aep.quantile(0.25), data_aep.quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
outlier_mask = (data_aep < lower) | (data_aep > upper)

daily_aep = df['AEP'].resample('D').mean()
ax4.plot(daily_aep.index, daily_aep.values, color='steelblue', linewidth=0.8, alpha=0.7, label='AEP (TB ngày)')
outlier_daily = df['AEP'][outlier_mask].resample('D').mean().dropna()
ax4.scatter(outlier_daily.index, outlier_daily.values,
            color='red', s=15, zorder=5, label=f'Ngày có outlier ({len(outlier_daily):,})')
ax4.axhline(upper, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Ngưỡng trên ({upper:,.0f})')
ax4.axhline(lower, color='orange', linestyle='--', linewidth=1, alpha=0.7, label=f'Ngưỡng dưới ({lower:,.0f})')
ax4.set_title('Chuỗi Thời Gian AEP & Các Điểm Bất Thường', fontweight='bold')
ax4.set_ylabel('MW')
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
ax4.legend(fontsize=8, ncol=2)

# --- 5. Phân phối residual (AEP - rolling mean 24h) ---
ax5 = fig.add_subplot(gs[2, :])
if 'AEP_roll_mean_24' in df.columns:
    residual = (df['AEP'] - df['AEP_roll_mean_24']).dropna()
    ax5.hist(residual, bins=100, color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.3)
    ax5.axvline(residual.mean(), color='black', linewidth=2, linestyle='--',
                label=f'Mean: {residual.mean():.1f}')
    ax5.axvline(residual.quantile(0.01), color='red', linewidth=1.5, linestyle=':',
                label=f'P1: {residual.quantile(0.01):,.0f}')
    ax5.axvline(residual.quantile(0.99), color='red', linewidth=1.5, linestyle=':',
                label=f'P99: {residual.quantile(0.99):,.0f}')
    ax5.set_title('Phân Phối Residual AEP (= AEP - Rolling Mean 24h)\nPhát hiện bất thường qua phân phối lệch', fontweight='bold')
    ax5.set_xlabel('Residual (MW)'); ax5.set_ylabel('Tần suất')
    ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
    ax5.legend(fontsize=10)
    pct_extreme = (abs(residual) > 2*residual.std()).sum() / len(residual) * 100
    ax5.text(0.98, 0.95, f'Điểm cực đoan (|r|>2σ): {pct_extreme:.2f}%',
             transform=ax5.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8), fontsize=10)

plt.savefig('output/charts/2_3_4_bat_thuong_chat_luong.png', bbox_inches='tight')
plt.close()
print("✅ Đã lưu: output/charts/2_3_4_bat_thuong_chat_luong.png")