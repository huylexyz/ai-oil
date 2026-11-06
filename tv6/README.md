# TV6 — XGBoost dự báo giá dầu Brent

## Phạm vi

Huấn luyện hai mô hình XGBoost dự báo trực tiếp giá dầu Brent
sau 1 phiên và 5 phiên quan sát, đơn vị USD/thùng.

Đầu vào sử dụng bộ đặc trưng do TV3 bàn giao:
`data/processed/brent_features.csv` (tính từ thư mục gốc repository `ai-oil`)..

Mỗi phiên tương ứng một dòng quan sát hợp lệ trong dữ liệu sạch,
không phải một ngày lịch.

## Dữ liệu

- Bộ đặc trưng: 2.933 dòng, 38 cột.
- Đầu vào mô hình: 29 đặc trưng.
- Khoảng ngày quan sát: 13/03/2015–30/09/2026.
- Không sử dụng observation_date, target_* hoặc split_* làm đầu vào.
- Dự báo được giả định thực hiện sau khi giá của phiên quan sát đã có.

## Chia tập và kiểm soát rò rỉ

Phân tập theo ngày mục tiêu:

- Train: đến hết năm 2022.
- Validation: năm 2023–2024.
- Test: 01/2025–09/2026.

Khi chọn tham số, chỉ giữ mẫu Train có ngày mục tiêu không sau
ngày quan sát đầu tiên của Validation.

Khi huấn luyện cuối, gộp Train + Validation và chỉ giữ mẫu có
ngày mục tiêu không sau ngày quan sát đầu tiên của Test.

Số mẫu:

| Giai đoạn | Horizon 1 | Horizon 5 |
|---|---:|---:|
| Train dùng tuning | 1.985 | 1.977 |
| Validation | 505 | 505 |
| Huấn luyện cuối | 2.490 | 2.482 |
| Test | 442 | 442 |

Mô hình cuối được giữ cố định trong giai đoạn Test.

## Chọn tham số

Thử 18 cấu hình cho mỗi horizon:

- max_depth: 2, 3, 5.
- learning_rate: 0.03, 0.05.
- min_child_weight: 3, 5, 10.

Các thiết lập cố định:

- objective: reg:squarederror.
- n_estimators tối đa khi tuning: 1000.
- early_stopping_rounds khi tuning: 50.
- subsample: 0.8.
- colsample_bytree: 0.8.
- reg_lambda: 1.0.
- tree_method: hist.
- random_state: 42.

Chọn cấu hình có RMSE Validation thấp nhất.
Không dùng Test để chọn tham số hoặc early stopping.

Cấu hình được chọn cho cả hai horizon:
max_depth=3, learning_rate=0.05, min_child_weight=5.

Số vòng boosting dùng khi huấn luyện cuối:
128 cho horizon 1 và 78 cho horizon 5.

## Kết quả Validation

Các chỉ số dưới đây thuộc mô hình của giai đoạn tuning,
học trên Train, chưa học lại trên Train + Validation.

| Horizon | Mô hình | MAE | RMSE |
|---|---|---:|---:|
| 1 | Naive | 1.1868 | 1.5542 |
| 1 | XGBoost | 1.2639 | 1.6185 |
| 5 | Naive | 2.6910 | 3.3764 |
| 5 | XGBoost | 2.8056 | 3.5276 |

XGBoost chưa vượt Naive trên Validation.
Không diễn giải các chỉ số này thành kết quả Test.

TV8 sử dụng dự báo Test để đánh giá MAE, RMSE và
Directional Accuracy theo quy ước chung của nhóm.

## Độ quan trọng đặc trưng

Đo bằng normalized gain từ mô hình cuối.

| Đặc trưng | Horizon 1 | Horizon 5 |
|---|---:|---:|
| ma_5 | 0.606010 | 0.385108 |
| lag_1 | 0.163895 | 0.310397 |
| price | 0.159512 | 0.154312 |

Các mô hình chủ yếu sử dụng nhóm thông tin mức giá gần hiện tại.
Importance không thể hiện chiều tác động, quan hệ nhân quả
hoặc trực tiếp đo độ chính xác trên Test.

Các đặc trưng tương quan có thể chia sẻ hoặc thay thế vai trò.
Importance bằng 0 chỉ cho biết đặc trưng không được dùng
trong các phép chia của mô hình đã huấn luyện.

## Cài đặt và chạy

Chạy từ thư mục gốc repository `C:\AI\ai-oil` bằng PowerShell.
Cần có file `data/processed/brent_features.csv` của TV3 ở đúng vị trí trước khi chạy.
Môi trường chung với TV4/TV5 đã được kiểm thử trên Python 3.14.5;
triển khai trên nền tảng khác cần kiểm tra lại phiên bản thư viện và artifact.
`main` hiện đã có dữ liệu TV2 và bộ 29 đặc trưng TV3 tại `data/processed/brent_features.csv`.

```powershell
cd C:\AI\ai-oil
python -m pip install -r requirements.txt -r tv6/requirements.txt
python tv6/check_data.py
python tv6/verify_handover.py
```

Nếu muốn huấn luyện lại (sẽ cập nhật artifact và các file kết quả), chạy:

```powershell
python tv6/tune_xgb.py
python tv6/finalize_xgb.py
python tv6/analyze_importance.py
python tv6/verify_handover.py
```

Chỉ thay đổi đường dẫn dữ liệu và hướng dẫn chạy; thuật toán, tham số,
model `.joblib` và kết quả dự báo gốc được giữ nguyên.
Trường `data_file` trong `results/xgb_summary.json` vẫn ghi lại đường dẫn
máy của TV6 tại lần huấn luyện gốc để giữ nguyên lịch sử thực nghiệm;
đó **không phải** đường dẫn mà code mới sử dụng. Hash CSV vẫn khớp.
Khi chạy lại `finalize_xgb.py`, trường này sẽ cập nhật theo workspace mới.

`train_xgb.py` là script baseline; thay HORIZON bằng 1 hoặc 5
để tái chạy bản thử nghiệm tương ứng. Không cần chạy baseline
để tạo mô hình cuối.

Chạy lại các script sẽ cập nhật các file kết quả tương ứng.

## Bàn giao cho TV7 — Dashboard

Mô hình:

- models/xgb_horizon_1.joblib
- models/xgb_horizon_5.joblib

Mỗi artifact là dictionary, có các khóa:
model, features, horizon, stage, params và metadata huấn luyện.

Ví dụ nạp mô hình, giả sử artifact_path là đường dẫn tới file
và input_df là DataFrame đặc trưng do pipeline TV3 tạo:

```python
artifact = joblib.load(artifact_path)
prediction = artifact["model"].predict(
    input_df[artifact["features"]]
)
```

Phải dùng đúng tên và thứ tự đặc trưng đã lưu.
Mô hình nhận đặc trưng, không nhận trực tiếp CSV giá thô.

Mỗi giá trị đầu ra là giá dự báo sau số phiên tương ứng.
Không cộng horizon theo ngày lịch để xác định ngày mục tiêu.

## Bàn giao cho TV8 — Đánh giá

File dự báo:

- results/xgb_predictions_test_h1.csv: 442 dòng.
- results/xgb_predictions_test_h5.csv: 442 dòng.
- results/xgb_predictions_test.csv: 884 dòng.

Các cột:
observation_date, target_date, horizon, actual, predicted, model, split.

Tất cả các dòng có split=test và model=XGBoost.
Khi tính Directional Accuracy, lấy giá tại observation_date
từ dữ liệu TV3 làm giá gốc để so sánh chiều biến động.

File bổ sung:

- results/tuning/: toàn bộ kết quả chọn tham số.
- results/xgb_summary.json: metadata, cấu hình, phiên bản và hash dữ liệu.
- results/xgb_importance_h1.csv và h5.csv.
- results/xgb_importance_h1.png và h5.png.

## Lưu ý tích hợp

Cần thống nhất với TV1/TV8 cách huấn luyện cuối và xử lý
ranh giới thời gian giữa các mô hình trước khi so sánh.

Kết quả Test chỉ phục vụ đánh giá cuối; không dùng để
quay lại lựa chọn cấu hình.

## Bổ sung horizon 2, 3 và 4

Chạy từ gốc repository trong môi trường chung đã cài `tv6/requirements.txt`:

```bash
python -m tv6.horizons_2_4
python -m unittest discover -s tests -p "test_*.py"
```

Script mới chỉ đọc `data/processed/brent_features_h{2,3,4}.csv` do TV3 bàn giao. Mỗi mô hình dùng đúng 29 cột `FEATURE_COLUMNS` theo thứ tự chính thức, `target_{h}` làm nhãn và `split_{h}` để chia tập. Giữ nguyên 18 cấu hình tìm kiếm, early stopping 50 vòng, seed 42 và các thiết lập cố định của TV6. Chọn cấu hình bằng Validation RMSE; mô hình cuối bỏ early stopping và dùng số vòng tốt nhất đã chọn. Test không tham gia tuning hoặc final fit.

Tại ranh giới Train/Validation, loại mẫu Train có `target_date_{h}` sau ngày quan sát Validation đầu tiên. Tại ranh giới Validation/Test, loại mẫu Validation có `target_date_{h}` sau ngày quan sát Test đầu tiên. Số mẫu loại tại mỗi ranh giới là 1/2/3 cho h2/h3/h4. Train dùng tuning: 1.984→1.983, 1.983→1.981, 1.982→1.979. Validation dùng final fit: 505→504, 505→503, 505→502. Fit cuối dùng 2.488/2.486/2.484 mẫu.

| Horizon | max_depth | learning_rate | min_child_weight | Số cây | Validation RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 0.03 | 10 | 176 | 2.288650 | 2.760181 | 4.207893 |
| 3 | 3 | 0.05 | 5 | 93 | 2.789931 | 3.593453 | 5.283781 |
| 4 | 2 | 0.03 | 3 | 127 | 3.150565 | 4.030534 | 6.078837 |

MAE/RMSE tính bằng USD/thùng. Trên Test, XGBoost h2 và h4 có MAE thấp nhất trong Naive, Ridge, Random Forest và XGBoost; XGBoost h3 có cả MAE và RMSE cao nhất. Không chọn lại tham số theo Test.

Mỗi horizon mới tạo `models/xgb_horizon_{h}.joblib`, `results/xgb_predictions_test_h{h}.csv`, `results/xgb_summary_h{h}.json` và bốn file tuning dưới `results/tuning/` theo quy ước h1/h5. Mỗi CSV dự báo có đúng 442 dòng Test. Cấu hình baseline cũ nằm trong 18 trial, nên không cần chạy riêng script baseline. Script mới từ chối ghi đè file đã có. Toàn bộ model, CSV, summary, bảng tuning và file gộp của h1/h5 giữ nguyên.
