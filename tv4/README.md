# TV4 — Naive Forecast & Ridge Regression

## 1. Phạm vi
TV4 triển khai Naive Forecast và Ridge Regression để dự báo giá dầu Brent sau 1 phiên và 5 phiên.

## 2. Cấu trúc thư mục

```text
Brent_TV4/
├── ../data/processed/brent_features.csv (do PR TV3 cung cấp)
├── models/
├── predictions/
├── naive_forecast.py
├── ridge_regression.py
├── requirements.txt
├── README.md
├── check_data.py
├── verify_outputs.py
└── verify_dates.py
```

## 3. Môi trường và cài đặt
Môi trường chung đã được kiểm thử trên Python 3.14.5; chưa xác nhận các phiên bản Python khác. Tạo môi trường ảo:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\\Scripts\\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Cài dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r tv4/requirements.txt
```

## 4. Dữ liệu đầu vào
Đọc dữ liệu TV3 tại `data/processed/brent_features.csv` ở gốc repository. `main` hiện đã có dữ liệu TV2 và bộ 29 đặc trưng TV3. Ridge sử dụng đúng 29 đặc trưng được liệt kê trong `ridge_regression.py`, cùng các cột nhãn và phân chia `target_1`, `target_5`, `target_date_1`, `target_date_5`, `split_1`, `split_5`.

Việc chia Train/Validation/Test được kế thừa từ dữ liệu TV3 và dựa trên ngày mục tiêu.

## 5. Chạy chương trình

```bash
python tv4/check_data.py
python tv4/naive_forecast.py
python tv4/ridge_regression.py
```

Kiểm tra đầu ra:

```bash
python tv4/verify_outputs.py
python tv4/verify_dates.py
```

## 6. Quy trình Ridge Regression

### Nguyên tắc chung
- Thử các giá trị `alpha`: `0.01`, `0.1`, `1`, `10`, `100`.
- Chọn alpha có MAE thấp nhất trên Validation.
- Mô hình dùng tạo dự báo Train/Validation chỉ được fit trên Train.
- Mô hình cuối dùng dự báo Test được fit lại trên Train + Validation hợp lệ.
- Test không tham gia chọn alpha hoặc huấn luyện mô hình.

### Horizon 1
1. Fit từng alpha trên Train.
2. Đánh giá trên Validation và chọn alpha có MAE thấp nhất.
3. Fit mô hình cuối với alpha đã chọn trên toàn bộ Train + Validation.
4. Trong `ridge_horizon_1.csv`, dự báo của Train và Validation được tạo bởi mô hình chỉ fit trên Train; dự báo Test được tạo bởi mô hình cuối fit trên Train + Validation.

### Horizon 5
**Khi tuning alpha:**
1. Xác định ngày quan sát đầu tiên của Validation.
2. Loại các mẫu Train có `target_date_5` sau ngày quan sát đầu tiên của Validation. Với dữ liệu hiện tại, có đúng 4 mẫu Train cuối bị loại.
3. Fit từng alpha trên phần Train còn lại và đánh giá trên toàn bộ Validation.
4. Chọn alpha có Validation MAE thấp nhất.

**Khi fit mô hình cuối:**
1. Ghép Train với Validation.
2. Xác định ngày quan sát đầu tiên của Test.
3. Trong Validation, loại các mẫu có `target_date_5` sau ngày quan sát đầu tiên của Test. Với dữ liệu hiện tại, có đúng 4 mẫu Validation cuối bị loại.
4. Fit mô hình cuối trên toàn bộ Train + phần Validation còn hợp lệ.
5. Trong `ridge_horizon_5.csv`, dự báo Train/Validation dùng mô hình tuning chỉ fit trên Train đã xử lý ranh giới; dự báo Test dùng mô hình cuối.

Chương trình kiểm tra số mẫu bị loại ở hai ranh giới Horizon 5. Nếu không đúng 4 mẫu theo dữ liệu hiện tại, chương trình dừng và yêu cầu kiểm tra lại dữ liệu TV3, thay vì âm thầm áp dụng sai quy tắc.

## 7. Sản phẩm đầu ra

- `models/ridge_horizon_1.joblib`
- `models/ridge_horizon_5.joblib`
- `predictions/naive_horizon_1.csv`
- `predictions/naive_horizon_5.csv`
- `predictions/ridge_horizon_1.csv`
- `predictions/ridge_horizon_5.csv`

Hai file Ridge `.joblib` lưu mô hình cuối, danh sách đặc trưng, horizon, alpha và thông tin quá trình huấn luyện. Horizon 5 lưu thêm ngày ranh giới và số dòng bị loại/được dùng ở từng bước.

Các CSV có các cột `observation_date`, `target_date`, `horizon`, `actual`, `predicted`, `model`, kèm `split` để TV8 lọc `split == test` khi đánh giá cuối cùng.

## 8. Naive Forecast
Naive hiện dùng `predicted = price` cho cả horizon 1 và 5. Giữ thống nhất định nghĩa này trong báo cáo nhóm.

## 9. Lưu ý đánh giá
Validation được dùng để chọn alpha. Test chỉ dùng để đánh giá cuối cùng; không dùng Test để chọn tham số hoặc huấn luyện mô hình. Khi TV8 đánh giá kết quả cuối cùng, lọc `split == test` trong từng file dự báo.

## 10. Bổ sung horizon 2, 3 và 4

Chạy từ gốc repository trong môi trường chung đã cài `tv4/requirements.txt`:

```bash
python -m tv4.horizons_2_4
python -m unittest discover -s tests -p "test_*.py"
```

Script mới chỉ đọc `data/processed/brent_features_h2.csv`, `brent_features_h3.csv` và `brent_features_h4.csv` do TV3 bàn giao. Đầu vào Ridge là đúng 29 cột `FEATURE_COLUMNS`, cùng thứ tự với `FEATURES` của TV4. Naive dùng `predicted = price` tại ngày quan sát. Các file dự báo mới có cùng bảy cột với CSV cũ nhưng **chỉ chứa 442 dòng Test** (`split=test`) cho mỗi mô hình/horizon; không tạo dự báo Train/Validation giả.

Ridge giữ nguyên pipeline `SimpleImputer → StandardScaler → Ridge` và tập alpha `0.01, 0.1, 1, 10, 100`. Alpha được chọn theo MAE trên Validation. Trước khi tuning, mẫu Train có ngày mục tiêu sau ngày quan sát Validation đầu tiên bị loại; trước final fit, mẫu Validation có ngày mục tiêu sau ngày quan sát Test đầu tiên bị loại. Số mẫu purge tại mỗi ranh giới là 1/2/3 tương ứng h2/h3/h4. Mô hình cuối fit trên Train cùng phần Validation hợp lệ, sau đó mới dự báo Test.

| Horizon | Alpha chọn | Naive Test MAE | Naive Test RMSE | Ridge Test MAE | Ridge Test RMSE |
|---|---:|---:|---:|---:|---:|
| 2 | 1 | 2.7798 | 4.2703 | 2.7785 | 4.2488 |
| 3 | 10 | 3.4797 | 5.2305 | 3.5225 | 5.2637 |
| 4 | 100 | 4.0442 | 5.9696 | 4.0668 | 6.1800 |

MAE/RMSE tính bằng USD/thùng. Trên Test, Ridge h3 và h4 kém Naive theo cả hai chỉ số; không dùng kết quả Test để chọn lại alpha.

Sản phẩm mới là `models/ridge_horizon_{2,3,4}.joblib`, `predictions/naive_horizon_{2,3,4}.csv` và `predictions/ridge_horizon_{2,3,4}.csv`. Script từ chối ghi đè file đã tồn tại. Sản phẩm horizon 1 và 5 tiếp tục dùng các script và file đã nghiệm thu ở trên.
