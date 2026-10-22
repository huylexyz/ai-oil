# Dự báo giá dầu Brent

Thực nghiệm dự báo giá Brent theo 1 và 5 phiên quan sát hợp lệ, với Naive Persistence, Ridge Regression, Random Forest và XGBoost. Giao diện kết quả dùng Streamlit.

## Chạy thực nghiệm

```powershell
python -m pip install -r requirements.txt
python -m src.experiment
python -m streamlit run app.py
```

`src.experiment` đọc số lượng quan sát hợp lệ hiện có, chạy đánh giá lịch sử và ghi:

- `results/metrics.csv`: MAE, RMSE, Directional Accuracy theo mô hình, horizon và tập. Directional Accuracy của Naive Persistence để trống vì mô hình dự báo giá không đổi, không dự báo hướng.
- `results/forecasts.csv`: dự báo cùng giá thực tế theo schema trong `src/config.py`.
- `results/data_quality.json`: chất lượng dữ liệu, số lượng quan sát từng tập và quy tắc tiền xử lý.
- `models/forecast_models.joblib`: 6 pipeline suy luận riêng cho Ridge Regression, Random Forest và XGBoost ở horizon 1 và 5. Ridge giữ nguyên `StandardScaler` trong pipeline.

Các kết quả backtest được lưu trong Git để dashboard có thể hiển thị ngay. Mô hình suy luận được fit sau khi đánh giá hoàn tất trên toàn bộ dữ liệu hợp lệ hiện có. Chạy lại `python -m src.experiment` sẽ tái tạo backtest và artifact.

## Dự báo mới và cập nhật dữ liệu

Chạy ứng dụng bằng `python -m streamlit run app.py`. Phần **Dự báo mới** tải artifact đã lưu, rồi tính giá Brent từ đặc trưng ở ngày quan sát cuối cùng. Chọn một trong ba mô hình và horizon 1 hoặc 5. Kết quả hiển thị USD/thùng, phần trăm thay đổi và hướng dự báo. Horizon là số phiên quan sát hợp lệ tiếp theo; ứng dụng không tự gán ngày tương lai.

Phần **Cập nhật dữ liệu và huấn luyện mô hình** nhận CSV có hai cột `observation_date` và `DCOILBRENTEU`. Giá trống hoặc không phải số được bỏ qua; ngày sai định dạng, giá không hữu hạn/không dương và các ngày trùng có mức giá xung đột sẽ bị từ chối. Quan sát trùng ngày cùng giá được gộp một lần. Sau khi kiểm tra, dữ liệu hợp nhất được lưu tại `data/processed/brent_observations.csv` và mô hình được fit lại tại `models/forecast_models.joblib`.

CSV nguồn trong `data/raw/` không bị sửa. Cập nhật dữ liệu không ghi vào `results/metrics.csv`, `results/forecasts.csv` hoặc `results/data_quality.json`; kết quả backtest lịch sử vẫn độc lập với artifact suy luận mới. Nút **Chạy lại backtest lịch sử** luôn đánh giá CSV gốc và giữ nguyên artifact suy luận hiện tại.

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
