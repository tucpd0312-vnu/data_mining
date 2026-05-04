import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('processed_energy_data.csv', index_col=0, parse_dates=True)

print("=" * 80)
print("THÔNG TIN CƠ BẢN VỀ TẬP DỮ LIỆU ĐÃ TIỀN XỬ LÝ")
print("=" * 80)

# 1. Kích thước
print(f"\n 1 KÍCH THƯỚC DỮ LIỆU:")
print(f"   - Số hàng (samples): {df.shape[0]:,}")
print(f"   - Số cột (features): {df.shape[1]:,}")
print(f"   - Khoảng thời gian: {df.index.min()} đến {df.index.max()}")
print(f"   - Tổng thời gian: {(df.index.max() - df.index.min()).days} ngày")

# 2. Missing values
print(f"\n 2️ MISSING VALUES:")
missing_count = df.isnull().sum().sum()
missing_pct = (missing_count / (df.shape[0] * df.shape[1])) * 100
print(f"   - Tổng giá trị thiếu: {missing_count:,} ({missing_pct:.2f}%)")
if missing_count > 0:
    print(f"   - Cột có missing values:")
    for col in df.columns[df.isnull().any()]:
        print(f"     • {col}: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.2f}%)")
else:
    print(f"Không có giá trị thiếu!")

# 3. Thống kê mô tả
print(f"\n 3 THỐNG KÊ MÔ TẢ:")
print(df.describe().round(2))

# 4. Loại cột
print(f"\n 4 PHÂN LOẠI CÁC CỘT:")
region_cols = [col for col in df.columns if any(region in col for region in 
    ['AEP', 'COMED', 'DAYTON', 'DEOK', 'DOM', 'DUQ', 'EKPC', 'FE', 'NI', 'PJM'])]
time_cols = [col for col in df.columns if col in 
    ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'weekofyear', 'is_weekend', 'is_holiday']]
lag_cols = [col for col in df.columns if 'lag_' in col]
rolling_cols = [col for col in df.columns if 'roll_' in col]
other_cols = [col for col in df.columns if col == 'total_MW']

print(f"   - Cột vùng địa lý (region): {len(region_cols)}")
print(f"   - Cột thời gian (time features): {len(time_cols)}")
print(f"   - Cột lag (quá khứ): {len(lag_cols)}")
print(f"   - Cột rolling statistics: {len(rolling_cols)}")
print(f"   - Cột khác: {len(other_cols)}")
print(f"\n   Chi tiết:")
print(f"      Region columns: {region_cols[:5]}...")
print(f"      Time columns: {time_cols}")
print(f"      Lag columns: {lag_cols}")

# 5. Data types
print(f"\n 5 KIỂU DỮ LIỆU:")
print(df.dtypes.value_counts())

print("\n Tất cả hoàn thành!")
