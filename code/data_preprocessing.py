import pandas as pd
import numpy as np
import os
from datetime import datetime
import holidays
from sklearn.impute import KNNImputer

# ====================== 1. ĐỌC DỮ LIỆU ======================
df = pd.read_csv('data/pjm_hourly_est.csv')
    
# Chuyển Datetime thành index
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime')
df = df.sort_index()

# Loại bỏ các thời điểm bị trùng lặp (nếu có, do chuyển đổi giờ mùa hè/mùa đông hoặc lỗi ghi nhận)
df = df[~df.index.duplicated(keep='first')]

print("Kích thước dataframe:", df.shape)
print("\nSố lượng giá trị thiếu mỗi khu vực:\n", df.isnull().sum())
print("\nTỉ lệ % thiếu mỗi khu vực:\n", (df.isnull().sum() / len(df) * 100).round(2))

# ====================== 2. XỬ LÝ MISSING VALUES ======================
df_processed = df.copy()

# Phương pháp 1: # Điền giá trị NaN bằng Trung bình có trọng số (dựa vào tương quan của 5 vùng cao nhất)
correlation_matrix = df.corr()
print("\n=== MA TRẬN TƯƠNG QUAN GIỮA CÁC VÙNG ===")
print(correlation_matrix.round(2))

for target_region in df_processed.columns:
    if target_region not in correlation_matrix.columns or df_processed[target_region].isnull().sum() == 0:
        continue
        
    # Lấy 5 vùng có độ tương quan dương cao nhất (bỏ qua chính nó)
    top_5_corr = correlation_matrix[target_region].nlargest(6)[1:]
    top_5_regions = top_5_corr.index.tolist()
    weights = top_5_corr.values
    
    # Tìm index các vị trí đang bị thiếu dữ liệu ở vùng hiện tại
    nan_mask = df_processed[target_region].isnull()
    
    if nan_mask.any():
        # Trích xuất dữ liệu của 5 vùng tương quan tại đúng các thời điểm bị missing
        support_data = df_processed.loc[nan_mask, top_5_regions].values
        
        # Bản đồ đánh dấu (mask) những giá trị hợp lệ, không bị NaN của các vùng lân cận
        valid_mask = ~np.isnan(support_data)
        
        # Tử số: Tổng của (Giá trị vùng lân cận * Độ tương quan)
        # dùng np.nansum để tự động bỏ qua các ô có giá trị NaN
        sum_weighted_values = np.nansum(support_data * weights, axis=1)
        
        # Mẫu số: Chỉ tính tổng trọng số của những vùng có dữ liệu thật sự
        sum_weights = np.sum(weights * valid_mask, axis=1)
        
        # Nội suy: Tử số / Mẫu số. 
        # Nếu mẫu số = 0 (tức là cả 5 vùng đều bị NaN vào giờ đó), thì giữ nguyên là NaN
        imputed_values = np.divide(sum_weighted_values, sum_weights, 
                                   out=np.full_like(sum_weighted_values, np.nan), 
                                   where=sum_weights!=0)
        
        # Gán lại giá trị đã nội suy
        df_processed.loc[nan_mask, target_region] = imputed_values.round(1)

# Phương pháp 2: Đánh dấu những dòng mà dữ liệu bị suy giảm (thêm feature cảnh báo)
# Điều này giúp mô hình biết phân biệt dữ liệu thật vs dữ liệu được điền
missing_rate = df.isnull().sum(axis=1) / df.shape[1]
df_processed['data_quality'] = (1 - missing_rate).values.round(1)

# Phương pháp 3: Chỉ giữ lại những dòng mà có ít nhất 50% dữ liệu thật
df_processed = df_processed[df.isnull().sum(axis=1) / df.shape[1] <= 0.5]

print(f"\nSau xử lý - Kích thước dataframe: {df_processed.shape}")
print(f"Giá trị NaN còn lại: {df_processed.isnull().sum().sum()}")

# Xóa cột NaN nếu có
df_processed = df_processed.dropna(axis=1, how='all')
df_processed.to_csv('processed_data.csv')

describe = df_processed.describe().round(2)
describe.to_csv('data_description.csv')

# Khởi tạo dict để chứa tất cả features
features_dict = {}

# Time features
features_dict['hour'] = df_processed.index.hour
features_dict['dayofweek'] = df_processed.index.dayofweek
features_dict['quarter'] = df_processed.index.quarter
features_dict['month'] = df_processed.index.month
features_dict['year'] = df_processed.index.year
features_dict['dayofyear'] = df_processed.index.dayofyear
features_dict['weekofyear'] = df_processed.index.isocalendar().week
features_dict['is_weekend'] = df_processed.index.dayofweek.isin([5,6]).astype(int)

# Holiday (US) - explicit cast để tránh FutureWarning
us_holidays = holidays.US(years=range(2002, 2019))
holiday_dates = pd.to_datetime(list(us_holidays.keys()))
features_dict['is_holiday'] = pd.to_datetime(df_processed.index.date).isin(holiday_dates).astype(int)

# Lag features
for lag in [1, 24, 168]:  # 1h trước, 1 ngày trước, 1 tuần trước
    for region in df_processed.columns:
        if region != 'data_quality':  # Bỏ qua feature data_quality
            features_dict[f'{region}_lag_{lag}'] = df_processed[region].shift(lag)

# Rolling statistics
for window in [24, 168]:  # 24h và 7 ngày
    for region in df_processed.columns:
        if region != 'data_quality':
            features_dict[f'{region}_roll_mean_{window}'] = df_processed[region].rolling(window).mean()
            features_dict[f'{region}_roll_std_{window}'] = df_processed[region].rolling(window).std()

# Aggregated features (tổng PJM)
energy_cols = [col for col in df_processed.columns if col != 'data_quality']
features_dict['total_MW'] = df_processed[energy_cols].sum(axis=1)

# Concat tất cả features một lần (tránh DataFrame fragmentation)
features_df = pd.DataFrame(features_dict, index=df_processed.index)
df_final = pd.concat([df_processed, features_df], axis=1)

print("Final shape with features:", df_final.shape)

# Xóa các cột NaN do lag và rolling
df_final = df_final.dropna(axis=1, how='all')
print(df_final.head())

# ====================== 4. LƯU FILE ======================
df_final.to_csv('processed_energy_feature_data.csv')

print("Hoàn thành tiền xử lý!")
