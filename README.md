# Dự báo giá dầu Brent

Thực nghiệm dự báo giá Brent theo 1 và 5 phiên quan sát hợp lệ, với Naive Persistence, Ridge Regression, Random Forest và XGBoost. Giao diện kết quả dùng Streamlit.

## Chạy thực nghiệm

```powershell
python -m pip install -r requirements.txt
python -m src.experiment
python -m streamlit run app.py
```

`src.experiment` kiểm tra kỳ vọng 2.982 quan sát hợp lệ, chạy pipeline và ghi:

- `results/metrics.csv`: MAE, RMSE, Directional Accuracy theo mô hình, horizon và tập. Directional Accuracy của Naive Persistence để trống vì mô hình dự báo giá không đổi, không dự báo hướng.
- `results/forecasts.csv`: dự báo cùng giá thực tế theo schema trong `src/config.py`.
- `results/data_quality.json`: chất lượng dữ liệu, số lượng quan sát từng tập và quy tắc tiền xử lý.

Các artifact kết quả được lưu trong Git để dashboard có thể hiển thị ngay. Có thể chạy lại từ nút trong ứng dụng hoặc từ dòng lệnh.

## Quy trình và quy ước

- Nguồn `data/raw/DCOILBRENTEU_2015-2026.csv` không bị sửa. Ngày không có giá số hợp lệ bị loại bỏ, không nội suy.
- Đặc trưng tại ngày dự báo gồm giá hiện tại và các độ trễ 1, 2, 3, 5, 10, 20 phiên; lợi suất trễ 1, 2, 5, 10, 20 phiên; trung bình và độ lệch chuẩn rolling 5, 10, 20 phiên. Tất cả rolling chỉ nhìn lùi và bao gồm giá tại ngày gốc.
- Horizon tính bằng số quan sát hợp lệ. Tập được gán theo ngày đích: Train đến 2022-12-31; Validation 2023-01-01–2024-12-31; Test 2025-01-01–2026-09-30.
- Validation dự báo từ mô hình fit trên Train. Test dự báo từ mô hình fit trên Train+Validation. Tham số được cố định trước; Test chỉ dùng để đánh giá cuối.
- Ridge dùng StandardScaler bên trong pipeline, fit riêng trên phần dữ liệu huấn luyện ở mỗi lượt. Mô hình cây không cần chuẩn hóa.
- Directional Accuracy so sánh dấu của thay đổi giá dự báo với thay đổi giá thực tế, cả hai tính từ giá tại ngày gốc.
- Naive Persistence lặp lại giá tại ngày gốc cho cả hai horizon. Đây là baseline cho sai số giá, không phải baseline dự báo hướng; DA không áp dụng.
- Horizon 1 và 5 dùng riêng mục tiêu tương lai tương ứng. Dữ liệu không bị xáo trộn.

## Dữ liệu và môi trường

CSV cần có cột `observation_date` và `DCOILBRENTEU`. Các thư viện được khai báo trong `requirements.txt`. `python scripts/check_project.py` kiểm tra cấu trúc tích hợp cơ bản.
