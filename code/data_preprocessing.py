import matplotlib
matplotlib.use('Agg')  # Không hiển thị cửa sổ, chỉ lưu file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
import os

# ====================== CẤU HÌNH ======================
OUTPUT_DIR = 'output/charts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style chung
sns.set_theme(style='whitegrid', palette='tab10')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'font.size': 11,
})

REGION_COLS = ['AEP', 'COMED', 'DAYTON', 'DEOK', 'DOM', 'DUQ', 'EKPC', 'FE', 'NI', 'PJME', 'PJMW']

# ====================== ĐỌC DỮ LIỆU ======================
print("Đang đọc dữ liệu...")
df = pd.read_csv('processed_energy_feature_data.csv', index_col=0, parse_dates=True)

# Chỉ lấy cột vùng thực có trong data
region_cols = [c for c in REGION_COLS if c in df.columns]

print(f"Dữ liệu: {df.shape[0]:,} hàng x {df.shape[1]} cột")
print(f"Thời gian: {df.index.min()} → {df.index.max()}")
print(f"Vùng: {region_cols}")
print()

# ====================== BIỂU ĐỒ 1: PHÂN PHỐI TẢI ĐIỆN THEO VÙNG (Boxplot) ======================
print("Vẽ biểu đồ 1: Boxplot phân phối theo vùng...")
fig, ax = plt.subplots(figsize=(14, 7))

data_box = [df[c].dropna().values for c in region_cols]
bp = ax.boxplot(data_box, labels=region_cols, patch_artist=True, notch=False,
                medianprops=dict(color='black', linewidth=2))

colors = sns.color_palette('tab10', len(region_cols))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title('Phân Phối Tải Điện Theo Vùng (MW)', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Vùng', fontsize=12)
ax.set_ylabel('Tải điện (MW)', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_boxplot_phan_phoi_theo_vung.png')
plt.close()
print(f"  → Đã lưu: {OUTPUT_DIR}/01_boxplot_phan_phoi_theo_vung.png")

# ====================== BIỂU ĐỒ 2: XU HƯỚNG TỔNG TẢI ĐIỆN THEO NĂM ======================
print("Vẽ biểu đồ 2: Xu hướng tổng tải điện theo năm...")
fig, ax = plt.subplots(figsize=(14, 5))

df_yearly = df['total_MW'].resample('ME').mean()
ax.plot(df_yearly.index, df_yearly.values, color='steelblue', linewidth=1.2, alpha=0.6, label='Trung bình tháng')

# Đường xu hướng năm
df_annual = df['total_MW'].resample('YE').mean()
ax.plot(df_annual.index, df_annual.values, color='crimson', linewidth=2.5,
        marker='o', markersize=6, label='Trung bình năm')

ax.set_title('Xu Hướng Tổng Tải Điện PJM Theo Thời Gian', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Năm', fontsize=12)
ax.set_ylabel('Tải điện (MW)', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_xu_huong_theo_nam.png')
plt.close()
print(f"  → Đã lưu: {OUTPUT_DIR}/02_xu_huong_theo_nam.png")

# ====================== BIỂU ĐỒ 3: MẪU TIÊU THỤ THEO GIỜ TRONG NGÀY ======================
print("Vẽ biểu đồ 3: Mẫu tiêu thụ theo giờ...")
fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharey=False)
axes = axes.flatten()

for i, col in enumerate(region_cols):
    ax = axes[i]
    hourly = df.groupby('hour')[col].mean()
    ax.plot(hourly.index, hourly.values, color=colors[i], linewidth=2.5, marker='o', markersize=4)
    ax.fill_between(hourly.index, hourly.values, alpha=0.15, color=colors[i])
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.set_xlabel('Giờ', fontsize=10)
    ax.set_ylabel('MW trung bình', fontsize=10)
    ax.set_xticks(range(0, 24, 4))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Tắt ô thừa
for j in range(len(region_cols), len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Mẫu Tiêu Thụ Điện Trung Bình Theo Giờ - Từng Vùng', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_mau_theo_gio.png')
plt.close()
print(f"  → Đã lưu: {OUTPUT_DIR}/03_mau_theo_gio.png")

# ====================== BIỂU ĐỒ 4: MẪU TIÊU THỤ THEO THÁNG ======================
print("Vẽ biểu đồ 4: Mẫu tiêu thụ theo tháng...")
fig, ax = plt.subplots(figsize=(14, 6))

month_labels = ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12']
for i, col in enumerate(region_cols):
    monthly = df.groupby('month')[col].mean()
    ax.plot(monthly.index, monthly.values, color=colors[i], linewidth=2,
            marker='o', markersize=5, label=col)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels)
ax.set_title('Mẫu Tiêu Thụ Điện Trung Bình Theo Tháng', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Tháng', fontsize=12)
ax.set_ylabel('MW trung bình', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.legend(loc='upper right', fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_mau_theo_thang.png')
plt.close()
print(f"  → Đã lưu: {OUTPUT_DIR}/04_mau_theo_thang.png")

# ====================== BIỂU ĐỒ 5: MẪU TIÊU THỤ THEO NGÀY TRONG TUẦN ======================
print("Vẽ biểu đồ 5: Mẫu tiêu thụ theo ngày trong tuần...")
fig, ax = plt.subplots(figsize=(12, 6))

day_labels = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'Chủ nhật']
for i, col in enumerate(region_cols):
    daily = df.groupby('dayofweek')[col].mean()
    ax.plot(daily.index, daily.values, color=colors[i], linewidth=2,
            marker='s', markersize=6, label=col)

ax.axvspan(4.5, 6.5, alpha=0.08, color='gray', label='Cuối tuần')
ax.set_xticks(range(7))
ax.set_xticklabels(day_labels)
ax.set_title('Mẫu Tiêu Thụ Điện Trung Bình Theo Ngày Trong Tuần', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Ngày', fontsize=12)
ax.set_ylabel('MW trung bình', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.legend(loc='upper right', fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_mau_theo_ngay_tuan.png')
plt.close()
print(f"  → Đã lưu: {OUTPUT_DIR}/05_mau_theo_ngay_tuan.png")

# ====================== BIỂU ĐỒ 6: SO SÁNH NGÀY THƯỜNG vs CUỐI TUẦN vs NGÀY LỄ ======================
print("Vẽ biểu đồ 6: So sánh ngày thường / cuối tuần / ngày lễ...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

labels = ['Ngày thường\n(is_weekend=0, is_holiday=0)',
          'Cuối tuần\n(is_weekend=1)',
          'Ngày lễ\n(is_holiday=1)']
masks = [
    (df['is_weekend'] == 0) & (df['is_holiday'] == 0),
    df['is_weekend'] == 1,
    df['is_holiday'] == 1,
]
palette = ['steelblue', 'coral', 'mediumseagreen']

# Dùng AEP làm đại diện, vẽ theo giờ
for ax, mask, label, color in zip(axes, masks, labels, palette):
    subset = df[mask]
    hourly = subset.groupby('hour')['AEP'].mean()
    hourly_std = subset.groupby('hour')['AEP'].std()
    ax.plot(hourly.index, hourly.values, color=color, linewidth=2.5, marker='o', markersize=4)
    ax.fill_between(hourly.index,
                    hourly.values - hourly_std.values,
                    hourly.values + hourly_std.values,
                    alpha=0.2, color=color)
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_xlabel('Giờ', fontsize=10)
    ax.set_ylabel('AEP (MW)', fontsize=10)
    ax.set_xticks(range(0, 24, 4))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

fig.suptitle('So Sánh Mẫu Tiêu Thụ: Ngày Thường / Cuối Tuần / Ngày Lễ (Vùng AEP)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_so_sanh_ngay_thuong_cuoi_tuan_le.png')
plt.close()
print(f"  → Đã lưu: {OUTPUT_DIR}/06_so_sanh_ngay_thuong_cuoi_tuan_le.png")

# ====================== BIỂU ĐỒ 7: HEATMAP TƯƠNG QUAN GIỮA CÁC VÙNG ======================
print("Vẽ biểu đồ 7: Heatmap tương quan...")
fig, ax = plt.subplots(figsize=(11, 9))

corr_matrix = df[region_cols].corr().round(2)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # Hiện cả ma trận
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            vmin=0.6, vmax=1.0, ax=ax,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Hệ số tương quan'})

ax.set_title('Ma Trận Tương Quan Giữa Các Vùng Điện', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_heatmap_tuong_quan.png')
plt.close()
print(f"  → Đã lưu: {OUTPUT_DIR}/07_heatmap_tuong_quan.png")

# ====================== BIỂU ĐỒ 8: HEATMAP GIỜ x THÁNG (tải điện trung bình AEP) ======================
print("Vẽ biểu đồ 8: Heatmap giờ x tháng...")
fig, ax = plt.subplots(figsize=(14, 7))

pivot = df.pivot_table(values='AEP', index='hour', columns='month', aggfunc='mean')
pivot.columns = ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12']

sns.heatmap(pivot, cmap='YlOrRd', annot=False, fmt='.0f', ax=ax,
            cbar_kws={'label': 'MW trung bình'},
            linewidths=0.3)

ax.set_title('Tải Điện Trung Bình AEP: Giờ × Tháng (MW)', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Tháng', fontsize=12)
ax.set_ylabel('Giờ trong ngày', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_heatmap_gio_x_thang.png')
plt.close()
print(f"  → Đã lưu: {OUTPUT_DIR}/08_heatmap_gio_x_thang.png")

# ====================== BIỂU ĐỒ 9: HISTOGRAM PHÂN PHỐI TẢI ĐIỆN ======================
print("Vẽ biểu đồ 9: Histogram phân phối...")
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(region_cols):
    ax = axes[i]
    data = df[col].dropna()
    ax.hist(data, bins=60, color=colors[i], alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.axvline(data.mean(), color='black', linewidth=1.8, linestyle='--', label=f'Mean: {data.mean():,.0f}')
    ax.axvline(data.median(), color='red', linewidth=1.8, linestyle=':', label=f'Median: {data.median():,.0f}')
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.set_xlabel('MW', fontsize=10)
    ax.set_ylabel('Tần suất', fontsize=10)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

for j in range(len(region_cols), len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Phân Phối Tần Suất Tải Điện - Từng Vùng', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_histogram_phan_phoi.png')
plt.close()
print(f"  → Đã lưu: {OUTPUT_DIR}/09_histogram_phan_phoi.png")

# ====================== BIỂU ĐỒ 10: CHUỖI THỜI GIAN 1 NĂM MẪU ======================
print("Vẽ biểu đồ 10: Chuỗi thời gian 1 năm mẫu (2017)...")
fig, ax = plt.subplots(figsize=(16, 5))

sample = df['2017']['total_MW'].resample('D').mean()
ax.plot(sample.index, sample.values, color='steelblue', linewidth=1.2)
ax.fill_between(sample.index, sample.values, sample.values.min(), alpha=0.15, color='steelblue')

ax.set_title('Tổng Tải Điện PJM - Trung Bình Ngày Năm 2017', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Thời gian', fontsize=12)
ax.set_ylabel('Tải điện (MW)', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/10_chuoi_thoi_gian_2017.png')
plt.close()
print(f"  → Đã lưu: {OUTPUT_DIR}/10_chuoi_thoi_gian_2017.png")

# ====================== TỔNG KẾT ======================
print()
print("=" * 55)
print("  HOÀN THÀNH! Tất cả biểu đồ đã lưu vào:")
print(f"  📁 {os.path.abspath(OUTPUT_DIR)}")
print("=" * 55)
print()
charts = sorted(os.listdir(OUTPUT_DIR))
for c in charts:
    print(f"   ✅ {c}")