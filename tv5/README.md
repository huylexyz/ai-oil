# TV5 – Random Forest dự báo giá dầu Brent (1 và 5 phiên)

## Cấu trúc
```
train_rf.py            # script huấn luyện (tune → fit lại → dự báo Test)
../data/processed/brent_features.csv  # đầu vào TV3 trong repository
requirements.txt
models/
  rf_model_t1.joblib, rf_model_t5.joblib            # mô hình cuối (RandomForestRegressor)
  rf_model_t1_metadata.json, rf_model_t5_metadata.json
results/
  rf_predictions_test_t1.csv, rf_predictions_test_t5.csv, rf_predictions_test.csv
  rf_tuning_h1.csv, rf_tuning_h5.csv                # kết quả tuning trên Validation
  rf_summary_t1.json, rf_summary_t5.json, rf_summary.json
```

## Cách chạy
Chạy từ gốc repository sau khi PR TV3 cung cấp `data/processed/brent_features.csv`.
Artifact bàn giao được tạo bằng Python 3.12.3, scikit-learn 1.8.0 và joblib 1.5.3.
Môi trường chung với TV4/TV6 đã được kiểm thử trên Python 3.14.5,
scikit-learn 1.8.0 và joblib 1.5.3; triển khai trên nền tảng khác cần kiểm tra lại.

```
python -m pip install -r requirements.txt -r tv5/requirements.txt
python tv5/train_rf.py
```
Chạy lại cho kết quả giống nhau (`random_state=42`) nếu cùng phiên bản thư viện (xem `library_versions` trong metadata).

## Quy trình
1. Đọc `brent_features.csv`, chia Train/Val/Test theo cột `split_1` / `split_5` (do TV3 chia theo `target_date`).
2. Đặc trưng = mọi cột số trừ `observation_date`, `target_*`, `target_date_*`, `split_*` (29 đặc trưng, chỉ dùng thông tin tại `observation_date`).
3. Tuning 6 tổ hợp tham số: fit trên Train, chọn theo **RMSE trên Validation**.
4. Fit lại tổ hợp tốt nhất trên **Train + Validation**, dự báo Test (Test không dùng để chọn tham số).

### Purge ranh giới (chỉ horizon 5)
Nhãn horizon 5 là giá sau 5 phiên nên cửa sổ nhãn của các mẫu cuối mỗi tập chồng sang tập kế tiếp. Quy tắc: loại mẫu ở tập trước nếu `target_date` > `observation_date` đầu tiên của tập sau.
- **Tuning:** loại 4 mẫu cuối của Train (quan sát 16, 19, 20, 21/12/2022) → fit trên 1.977 mẫu thay vì 1.981. Validation giữ nguyên 505 mẫu.
- **Fit cuối:** loại 4 mẫu cuối của Validation (quan sát 16–20/12/2024) → fit trên 2.482 mẫu. Test giữ nguyên 442 mẫu.
- Danh sách ngày bị loại nằm trong `purged_boundary_samples` của metadata và summary.
- Horizon 1 giữ nguyên, không purge (`PURGE_HORIZONS = {5}` trong `train_rf.py`).

## Định dạng file dự báo
`observation_date, target_date, horizon, actual, predicted, model, split` — `split` luôn là `test`. `rf_predictions_test.csv` gộp cả hai horizon (884 dòng).

## Metadata mô hình (`models/rf_model_t{h}_metadata.json`)
`horizon`, `target_column`, `features` (đúng thứ tự cột lúc huấn luyện), `best_params`, chỉ số Validation, số mẫu từng tập, số mẫu purge, khoảng ngày dữ liệu fit, phiên bản thư viện, thời điểm tạo.

Nạp mô hình:
```python
import joblib, json
meta = json.load(open("models/rf_model_t5_metadata.json"))
model = joblib.load(meta["model_file"])
y = model.predict(X[meta["features"]])   # giữ đúng thứ tự cột
```

## Kết quả Validation (chọn tham số)
| Horizon | Tham số tốt nhất | MAE | RMSE |
|---|---|---|---|
| 1 | n=300, depth=None, leaf=2, max_features=0.8 | 1.2987 | 1.6880 |
| 5 | n=200, depth=None, leaf=2, max_features=1.0 | 2.9466 | 3.7757 |

Chỉ số Test do TV8 đánh giá độc lập từ các file predictions.

## Bổ sung horizon 2, 3 và 4

Chạy từ gốc repository trong môi trường chung đã cài `tv5/requirements.txt`:

```bash
python -m tv5.horizons_2_4
python -m unittest discover -s tests -p "test_*.py"
```

Script mới chỉ đọc ba file `data/processed/brent_features_h{2,3,4}.csv` của TV3. Đầu vào mô hình giữ đúng 29 cột `FEATURE_COLUMNS` theo thứ tự chính thức; nhãn là `target_{h}`, tập dữ liệu lấy từ `split_{h}`. Giữ nguyên 6 cấu hình `PARAM_GRID`, `random_state=42`, và chọn tham số bằng **Validation RMSE**. Test chỉ được dùng sau khi chọn tham số và fit cuối.

Trước tuning, loại mẫu Train có `target_date_{h}` sau `observation_date` đầu tiên của Validation. Trước final fit, loại mẫu Validation có `target_date_{h}` sau `observation_date` đầu tiên của Test. Tương ứng h2/h3/h4, mỗi ranh giới loại 1/2/3 mẫu. Final fit dùng toàn bộ Train và phần Validation còn hợp lệ, giống quy trình TV5 h5. Số mẫu Train trước/sau purge là 1.984/1.983 (h2), 1.983/1.981 (h3), 1.982/1.979 (h4). Số mẫu fit cuối là 2.488, 2.486, 2.484.

Mỗi horizon mới có `models/rf_model_t{h}.joblib`, `models/rf_model_t{h}_metadata.json`, `results/rf_predictions_test_t{h}.csv`, `results/rf_tuning_h{h}.csv` và `results/rf_summary_t{h}.json`. Mỗi CSV dự báo có đúng 442 dòng Test theo schema cũ. Script từ chối ghi đè sản phẩm đã tồn tại. File gộp `results/rf_predictions_test.csv` và mọi sản phẩm h1/h5 giữ nguyên.

| Horizon | n_estimators | max_depth | min_samples_leaf | max_features | Test MAE | Test RMSE |
|---|---:|---|---:|---:|---:|---:|
| 2 | 300 | None | 2 | 0.8 | 2.8054 | 4.1433 |
| 3 | 300 | None | 2 | 0.8 | 3.5612 | 5.0009 |
| 4 | 300 | None | 2 | 0.8 | 4.1758 | 5.7124 |

MAE/RMSE tính bằng USD/thùng. Trên Test, MAE của Random Forest h2/h3/h4 cao hơn Naive và Ridge tương ứng, còn RMSE thấp hơn cả hai. Không chọn lại tham số theo kết quả Test.
