import pandas as pd
import numpy as np
import os
from datetime import datetime
import holidays

# ====================== 1. MERGE 14 FILE ======================
data_dir = 'D:/Personal/1_Master/Data_mining/project/data'

csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
print(f"Tìm thấy {len(csv_files)} file: {csv_files}")

dfs = []
for file in csv_files:
    df = pd.read_csv(os.path.join(data_dir, file))
    region_name = file.replace('_hourly.csv', '')
    # Đổi tên cột MW thành tên region
    mw_col = [col for col in df.columns if col != 'Datetime'][0]
    df = df.rename(columns={mw_col: region_name})
    dfs.append(df)
    
# Merge theo Datetime
merged = dfs[0]
for df in dfs[1:]:
    merged = pd.merge(merged, df, on='Datetime', how='outer')
    
# Chuyển Datetime thành index
merged['Datetime'] = pd.to_datetime(merged['Datetime'])
merged = merged.set_index('Datetime')
merged = merged.sort_index()

print("Kích thước dataframe sau merge:", merged.shape)
print("\nSố lượng giá trị thiếu mỗi khu vực:\n", merged.isnull().sum())

# ====================== 2. XỬ LÝ MISSING VALUES ======================
# Forward + Backward fill
merged = merged.ffill().bfill()

print("Giá trị thiếu sau xử lý:", merged.isnull().sum().sum())

df = merged.copy()

# Khởi tạo dict để chứa tất cả features
features_dict = {}

# Time features
features_dict['hour'] = df.index.hour
features_dict['dayofweek'] = df.index.dayofweek
features_dict['quarter'] = df.index.quarter
features_dict['month'] = df.index.month
features_dict['year'] = df.index.year
features_dict['dayofyear'] = df.index.dayofyear
features_dict['weekofyear'] = df.index.isocalendar().week
features_dict['is_weekend'] = df.index.dayofweek.isin([5,6]).astype(int)

# Holiday (US) - explicit cast để tránh FutureWarning
us_holidays = holidays.US(years=range(2002, 2019))
holiday_dates = pd.to_datetime(list(us_holidays.keys()))
features_dict['is_holiday'] = pd.to_datetime(df.index.date).isin(holiday_dates).astype(int)

# Lag features
for lag in [1, 24, 168]:  # 1h trước, 1 ngày trước, 1 tuần trước
    for region in merged.columns:
        features_dict[f'{region}_lag_{lag}'] = merged[region].shift(lag)

# Rolling statistics
for window in [24, 168]:  # 24h và 7 ngày
    for region in merged.columns:
        features_dict[f'{region}_roll_mean_{window}'] = merged[region].rolling(window).mean()
        features_dict[f'{region}_roll_std_{window}'] = merged[region].rolling(window).std()

# Aggregated features (tổng PJM)
features_dict['total_MW'] = merged.sum(axis=1)

# Concat tất cả features một lần (tránh DataFrame fragmentation)
df = pd.concat([df, pd.DataFrame(features_dict)], axis=1)

print("Final shape with features:", df.shape)

# Xóa các cột NaN do lag và rolling
df = df.dropna()
print(df.head())

# ====================== 4. LƯU FILE ======================
df.to_csv('processed_energy_data.csv')
merged.to_csv('merged_raw_14regions.csv')

print("Hoàn thành tiền xử lý!")
