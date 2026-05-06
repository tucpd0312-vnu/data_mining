"""
RAW_05. So Sánh Trước vs Sau Xử Lý
Output: output/charts/raw_05_truoc_sau_xu_ly.png
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
df_raw = pd.read_csv('data/pjm_hourly_est.csv')
df_raw['Datetime'] = pd.to_datetime(df_raw['Datetime'])
df_raw = df_raw.set_index('Datetime').sort_index()
df_raw = df_raw[~df_raw.index.duplicated(keep='first')]

df_proc = pd.read_csv('processed_energy_feature_data.csv', index_col=0, parse_dates=True)

region_cols = [c for c in ['AEP','COMED','DAYTON','DEOK','DOM','DUQ','EKPC','FE','NI','PJME','PJMW']
               if c in df_raw.columns and c in df_proc.columns]
colors = sns.color_palette('tab10', len(region_cols))

fig = plt.figure(figsize=(20, 20))
fig.suptitle('RAW 05 — So Sánh Dữ Liệu Trước & Sau Xử Lý', fontsize=18, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

# --- 1. Missing rate: trước vs sau ---
ax1 = fig.add_subplot(gs[0, :])
missing_before = (df_raw[region_cols].isnull().sum() / len(df_raw) * 100)
missing_after  = (df_proc[region_cols].isnull().sum() / len(df_proc) * 100)
x = np.arange(len(region_cols))
w = 0.35
bars1 = ax1.bar(x - w/2, missing_before.values, w, label='Trước xử lý',
                color='#d62728', alpha=0.8, edgecolor='white')
bars2 = ax1.bar(x + w/2, missing_after.values,  w, label='Sau xử lý',
                color='#2ca02c', alpha=0.8, edgecolor='white')
ax1.set_xticks(x); ax1.set_xticklabels(region_cols)
ax1.set_title('Tỷ Lệ Missing Values: Trước vs Sau Xử Lý (%)', fontweight='bold')
ax1.set_ylabel('% dữ liệu thiếu')
ax1.legend(fontsize=11)
for bar, val in zip(bars1, missing_before.values):
    if val > 0.5:
        ax1.text(bar.get_x()+bar.get_width()/2, val+0.5, f'{val:.1f}%',
                 ha='center', va='bottom', fontsize=8, color='#d62728')
for bar, val in zip(bars2, missing_after.values):
    ax1.text(bar.get_x()+bar.get_width()/2, val+0.5, f'{val:.2f}%',
             ha='center', va='bottom', fontsize=8, color='#2ca02c')

# --- 2. Số hàng & cột: trước vs sau ---
ax2 = fig.add_subplot(gs[1, 0])
categories = ['Số hàng (nghìn)', 'Số cột']
vals_before = [len(df_raw)/1000, df_raw.shape[1]]
vals_after  = [len(df_proc)/1000, df_proc.shape[1]]
x2 = np.arange(len(categories))
ax2.bar(x2 - 0.2, vals_before, 0.35, label='Trước xử lý', color='#d62728', alpha=0.8)
ax2.bar(x2 + 0.2, vals_after,  0.35, label='Sau xử lý',   color='#2ca02c', alpha=0.8)
ax2.set_xticks(x2); ax2.set_xticklabels(categories)
ax2.set_title('Kích Thước Dữ Liệu\nTrước vs Sau Xử Lý', fontweight='bold')
ax2.legend(fontsize=10)
for val, x_pos in [(vals_before[0], x2[0]-0.2), (vals_before[1], x2[1]-0.2),
                    (vals_after[0],  x2[0]+0.2), (vals_after[1],  x2[1]+0.2)]:
    ax2.text(x_pos, val + max(vals_after)*0.02,
             f'{val:.0f}K' if val > 10 else f'{int(val)}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# --- 3. So sánh phân phối AEP: trước vs sau ---
ax3 = fig.add_subplot(gs[1, 1])
raw_aep  = df_raw['AEP'].dropna()
proc_aep = df_proc['AEP'].dropna()
ax3.hist(raw_aep,  bins=60, alpha=0.55, color='#d62728', label=f'Trước (n={len(raw_aep):,})', density=True)
ax3.hist(proc_aep, bins=60, alpha=0.55, color='#2ca02c', label=f'Sau (n={len(proc_aep):,})',  density=True)
ax3.axvline(raw_aep.mean(),  color='#d62728', linewidth=2, linestyle='--', label=f'Mean trước: {raw_aep.mean():,.0f}')
ax3.axvline(proc_aep.mean(), color='#2ca02c', linewidth=2, linestyle='--', label=f'Mean sau: {proc_aep.mean():,.0f}')
ax3.set_title('Phân Phối AEP\nTrước vs Sau Xử Lý (Density)', fontweight='bold')
ax3.set_xlabel('MW'); ax3.set_ylabel('Density')
ax3.legend(fontsize=8, ncol=2)

# --- 4. So sánh chuỗi thời gian AEP ---
ax4 = fig.add_subplot(gs[2, :])
# Chỉ lấy khoảng overlap
common_start = max(df_raw.index.min(), df_proc.index.min())
common_end   = min(df_raw.index.max(), df_proc.index.max())
raw_ts  = df_raw.loc[common_start:common_end, 'AEP'].resample('W').mean()
proc_ts = df_proc.loc[common_start:common_end, 'AEP'].resample('W').mean()

ax4.plot(raw_ts.index,  raw_ts.values,  color='#d62728', linewidth=1,   alpha=0.7,
         label='AEP gốc (TB tuần)')
ax4.plot(proc_ts.index, proc_ts.values, color='#2ca02c', linewidth=1.5, alpha=0.85,
         label='AEP đã xử lý (TB tuần)', linestyle='--')

# Highlight khoảng NaN gốc
nan_mask_raw = df_raw['AEP'].resample('W').apply(lambda x: x.isna().all())
for idx in nan_mask_raw[nan_mask_raw].index:
    ax4.axvspan(idx - pd.Timedelta(weeks=1), idx, alpha=0.15, color='gray')

ax4.set_title('Chuỗi Thời Gian AEP: Trước vs Sau Xử Lý (TB tuần)\nVùng xám = tuần thiếu dữ liệu gốc', fontweight='bold')
ax4.set_ylabel('MW')
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax4.legend(fontsize=10, ncol=2)

# Thêm text box tổng kết
textstr = (
    f"Tóm tắt:\n"
    f"• Hàng gốc: {len(df_raw):,} → Sau xử lý: {len(df_proc):,}\n"
    f"• Cột gốc: {df_raw.shape[1]} → Sau feature engineering: {df_proc.shape[1]}\n"
    f"• Missing AEP: {missing_before['AEP']:.1f}% → {missing_after['AEP']:.2f}%\n"
    f"• Mean AEP: {raw_aep.mean():,.0f} MW → {proc_aep.mean():,.0f} MW"
)
ax4.text(0.01, 0.97, textstr, transform=ax4.transAxes,
         fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.savefig('output/charts/raw_05_truoc_sau_xu_ly.png', bbox_inches='tight')
plt.close()
print("✅ Đã lưu: output/charts/raw_05_truoc_sau_xu_ly.png")