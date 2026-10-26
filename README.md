# Dự báo biến động giá dầu Brent

Dự án nghiên cứu ứng dụng Machine Learning để dự báo giá dầu Brent theo dữ liệu lịch sử, với thời hạn dự báo 1 và 5 phiên. Các mô hình dự kiến gồm Naive Forecast, Ridge Regression, Random Forest và XGBoost; giao diện dùng Streamlit và Plotly.

## Trạng thái bàn giao

Repository hiện có dữ liệu giá lịch sử trong `data/raw/`. Các mô-đun huấn luyện, dự báo và đánh giá mô hình chưa được bàn giao; ứng dụng không tạo hoặc hiển thị dự báo hay chỉ số thực nghiệm giả. Hãy chạy `python scripts/check_project.py` để xem trạng thái tích hợp hiện tại.

## Cài đặt và chạy

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/check_project.py
python -m compileall -q app.py src scripts
python -m streamlit run app.py
```

Ứng dụng hiển thị chuỗi giá lịch sử khi dữ liệu khả dụng. Tính năng dự báo và bảng đánh giá sẽ khả dụng sau khi nhóm bàn giao các mô-đun và artifact mô hình tương ứng.

## Dữ liệu

Đặt CSV trong `data/raw/`. Mặc định ứng dụng đọc `data/raw/DCOILBRENTEU_2015-2026.csv`, với cột ngày `observation_date` và cột giá `DCOILBRENTEU`. Dữ liệu thiếu được bỏ qua khi vẽ lịch sử. Không có dữ liệu hay kết quả mô phỏng được tạo làm dữ liệu nghiên cứu.

Quy trình dữ liệu TV2/TV3 có thể chạy lại bằng `python scripts/clean_data.py` rồi `python -m src.features`. Kết quả lần lượt được lưu tại `data/processed/brent_clean.csv` và `data/processed/brent_features.csv`; báo cáo làm sạch nằm trong `docs/`. Notebook EDA đã chuẩn hóa đường dẫn cho cấu trúc repo tại `notebooks/01_EDA.ipynb`.

## Quy ước tích hợp

- Module Python đặt trong `src/`; đường dẫn dữ liệu và cấu hình tập trung trong `src/config.py`.
- Horizon dùng số phiên giao dịch: `1` hoặc `5`.
- Bảng dự báo/đánh giá dùng cùng các cột `model`, `horizon`, `origin_date`, `target_date`, `actual`, `predicted`, `split`.
- Train, validation và test phải được chia theo thời gian, không xáo trộn thứ tự quan sát.
- Mỗi thành viên làm trên nhánh riêng, mở Pull Request vào `main`; yêu cầu review trước khi merge. Không push trực tiếp lên `main`.

## Cấu trúc

```text
data/{raw,processed}/  notebooks/  src/  models/  results/
reports/  slides/  docs/  scripts/  app.py
```
