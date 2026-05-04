# Bài toán Dự báo Nhu cầu năng lượng

## Tổng Quan Dự Án

Dự án này nhằm xây dựng mô hình máy học để dự báo tải điện (energy load) hàng giờ tại các vùng khác nhau ở Hoa Kỳ. Dữ liệu bao gồm 14 vùng khác nhau với dữ liệu tiêu thụ điện hàng giờ từ năm 1998 đến 2018.

Dự án được thiết kế theo quy trình tiêu chuẩn trong học máy: tải dữ liệu, tiền xử lý dữ liệu, khai phá dữ liệu và xây dựng mô hình học máy.

## Cấu Trúc Thư Mục

```
project/
├── code/
│   ├── download_data.py              - Tải dữ liệu từ Kaggle
│   ├── data_preprocessing.py         - Tiền xử lý và tạo features
│   ├── explore_processed_data.py     - Phân tích và visualize dữ liệu
├── data/
│   ├── merged_raw_14regions.csv      - Dữ liệu gốc sau merge (không có features)
│   ├── processed_energy_data.csv     - Dữ liệu sau xử lý (đầu vào cho model)
│   ├── AEP_hourly.csv
│   ├── COMED_hourly.csv
│   ├── DAYTON_hourly.csv
│   ├── DEOK_hourly.csv
│   ├── DOM_hourly.csv
│   ├── DUQ_hourly.csv
│   ├── EKPC_hourly.csv
│   ├── FE_hourly.csv
│   ├── NI_hourly.csv
│   ├── PJME_hourly.csv
│   ├── PJMW_hourly.csv
│   ├── PJM_Load_hourly.csv
│   └── ... (các file dữ liệu tiêu thụ điện hàng giờ của mỗi vùng)
├── requirements.txt                  - Các thư viện Python cần thiết
└── README.md               
```

## Mô Tả Chi Tiết Các File Code

### 1. download_data.py

Chức năng: Tải bộ dữ liệu tiêu thụ điện hàng giờ từ Kaggle.

Nội dung:

- Sử dụng thư viện kagglehub để kết nối và tải bộ dữ liệu "hourly-energy-consumption"
- Dữ liệu sẽ được tải vào thư mục data

Lưu ý: Cần cấu hình API key cho Kaggle trước khi chạy lệnh này. Hãy tham khảo hướng dẫn chính thức của Kaggle để thiết lập.

### 2. data_preprocessing.py

Chức năng: Tiền xử lý dữ liệu thô và tạo các features mới cho mô hình học máy.

Quy trình chi tiết:

#### Bước 1: Merge 14 File Dữ Liệu

Công việc:

- Quét thư mục data để tìm tất cả file CSV
- Đọc từng file vào pandas DataFrame
- Đổi tên cột MW thành tên vùng (AEP, COMED, v.v.) để dễ nhận biết
- Merge tất cả 14 DataFrame theo cột Datetime bằng outer join (giữ toàn bộ dữ liệu)
- Chuyển Datetime thành index và sắp xếp theo thời gian

Phương pháp: Sử dụng pd.merge với outer join để đảm bảo không mất dữ liệu

--> Dataframe với 14 cột các vùng và chỉ số thời gian

#### Bước 2: Xử Lý Giá Trị Thiếu

Công việc:

- Kiểm tra số lượng giá trị NaN (Not a Number) trong mỗi vùng
- Áp dụng Forward Fill (ffill): Lấy giá trị hàng trước để điền vào ô thiếu
- Áp dụng Backward Fill (bfill): Nếu vẫn còn NaN ở đầu dữ liệu, lấy giá trị hàng sau

Lý do chọn phương pháp này:

- Forward fill phù hợp với dữ liệu chuỗi thời gian vì tải điện thường thay đổi từ từ
- Backward fill là backup cho những giá trị NaN ở đầu
- Phương pháp đơn giản nhưng hiệu quả cho dữ liệu năng lượng

#### Bước 3: Tạo Các Features Mới (Feature Engineering)

Đây là bước quan trọng vì mô hình học máy sẽ học từ các đặc trưng này.

A. Features Thời Gian:

- hour: Giờ trong ngày (0-23) - để học mẫu tiêu thụ theo giờ
- dayofweek: Ngày trong tuần (0-6) - để học mẫu theo ngày
- quarter: Quý trong năm (1-4) - để học mẫu theo mùa
- month: Tháng (1-12) - để học ảnh hưởng từng tháng
- year: Năm (2002-2018) - để học xu hướng dài hạn
- dayofyear: Ngày trong năm (1-365) - để học mẫu hàng năm
- weekofyear: Tuần trong năm (1-53) - để học mẫu theo tuần
- is_weekend: Cờ cuối tuần (0 hoặc 1) - phân biệt ngày thường và cuối tuần

B. Features Ngày Lễ:

- is_holiday: Cờ ngày lễ Hoa Kỳ (0 hoặc 1) - tải điện thường giảm vào ngày lễ

Phương pháp: Sử dụng thư viện holidays để xác định chính xác ngày lễ Mỹ

C. Lag Features (Dữ Liệu Quá Khứ):

- region_lag_1: Tải điện 1 giờ trước
- region_lag_24: Tải điện 1 ngày trước (cùng giờ hôm qua)
- region_lag_168: Tải điện 1 tuần trước (cùng giờ tuần trước)

Lý do: Tải điện thường có mối liên hệ mạnh với giá trị gần đây, giúp mô hình dự báo tốt hơn

D. Rolling Statistics (Thống Kê Trượt):

- region_roll_mean_24: Trung bình tải điện trong 24 giờ qua
- region_roll_std_24: Độ lệch chuẩn trong 24 giờ qua
- region_roll_mean_168: Trung bình tải điện trong 7 ngày qua
- region_roll_std_168: Độ lệch chuẩn trong 7 ngày qua

Lý do: Capture xu hướng ngắn hạn (24h) và trung hạn (7 ngày) của tải điện

E. Features Tổng Hợp:

- total_MW: Tổng tải điện của tất cả 14 vùng

Lý do: Tải điện toàn hệ thống có thể ảnh hưởng đến tải từng vùng

Phương pháp tạo features:

- Tạo tất cả features trong một dictionary để tối ưu hiệu suất
- Sử dụng pd.concat để thêm một lúc thay vì thêm từng cột
- Xóa các hàng và cột chứa NaN do lag và rolling

#### Bước 4: Lưu Dữ Liệu

Hai file CSV được lưu:

- processed_energy_data.csv: Dữ liệu đầy đủ với tất cả features (dùng cho mô hình ML)
- merged_raw_14regions.csv: Dữ liệu gốc sau merge mà không có features (để tham chiếu)

### 3. explore_processed_data.py

Chức năng: Phân tích cơ bản dữ liệu đã tiền xử lý để hiểu rõ hơn về bộ dữ liệu.

#### Nội dung:

1. Kích thước dữ liệu:

- Số hàng (samples): Bao nhiêu điểm dữ liệu
- Số cột (features): Bao nhiêu đặc trưng
- Khoảng thời gian: Từ ngày nào đến ngày nào
- Tổng thời gian: Bao nhiêu ngày

2. Kiểm tra giá trị thiếu:

- Tổng số giá trị NaN và tỷ lệ phần trăm
- Chi tiết từng cột có bao nhiêu giá trị thiếu

3. Thống kê mô tả:

- Mean (trung bình), std (độ lệch chuẩn)
- Min, 25%, 50%, 75%, max (các phân vị)
- Giúp hiểu phân bố của dữ liệu

4. Phân loại các cột:

- Bao nhiêu cột region (vùng địa lý)
- Bao nhiêu cột time features (thời gian)
- Bao nhiêu cột lag (quá khứ)
- Bao nhiêu cột rolling statistics

## Quy Trình Chạy Dự Án

Để chạy dự án một cách hoàn chỉnh, thực hiện theo thứ tự sau:

Bước 1: Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

Bước 2: Tải dữ liệu từ Kaggle (nếu chưa có)

```bash
python download_data.py
```

Bước 3: Tiền xử lý dữ liệu và tạo features

```bash
python data_preprocessing.py
```

Bước 4: Khai phá và phân tích dữ liệu

```bash
python explore_processed_data.py
```

Sau khi hoàn thành, thu được:

- processed_energy_data.csv: Dữ liệu để visualize, phân tích thống kê, ... và huấn luyện các mô hình học máy
- merged_raw_14regions.csv: Dữ liệu gốc để reference

## Các Thư Viện Chính

- pandas: Xử lý dữ liệu và DataFrame
- numpy: Tính toán số học
- matplotlib: Vẽ biểu đồ
- seaborn: Tạo biểu đồ thống kê đẹp hơn
- holidays: Xác định ngày lễ
- kagglehub: Tải dữ liệu từ Kaggle
