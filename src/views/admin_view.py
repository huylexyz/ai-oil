import json
import math

import pandas as pd
import streamlit as st

from src.config import PROJECT_ROOT
from src.backtest_data import COLUMNS, EXPECTED_TEST_ROWS, HORIZONS, SOURCES
from src.core.config import FEATURES_DATA_PATH, MODEL_FILE_CANDIDATES
from src.services.data_service import (
    get_clean_data,
    get_removed_rows,
    get_features_data,
    get_all_predictions,
    load_trained_model,
)
from src.ui.components import (
    render_section_header,
    render_bento_kpis,
    render_genesis_table,
    render_editorial_note,
)


def render_admin_view():
    try:
        clean_df, has_clean = get_clean_data()
        if not has_clean or not isinstance(clean_df, pd.DataFrame) or clean_df.empty:
            raise ValueError("Chưa tìm thấy bộ dữ liệu lịch sử Brent đã làm sạch.")
        required = {"observation_date", "price"}
        missing = required.difference(clean_df.columns)
        if missing:
            raise ValueError(f"Dữ liệu Brent thiếu cột bắt buộc: {sorted(missing)}")
        dates = pd.to_datetime(clean_df["observation_date"], errors="raise")
        prices = pd.to_numeric(clean_df["price"], errors="raise")
        if dates.isna().any() or dates.duplicated().any() or prices.isna().any():
            raise ValueError("Dữ liệu Brent có ngày hoặc giá bị thiếu hay trùng lặp.")
        if not prices.map(math.isfinite).all() or not prices.gt(0).all():
            raise ValueError("Giá Brent phải là số hữu hạn và dương.")
    except Exception as exc:
        st.error(f"Không thể hiển thị dữ liệu Brent cho trang Quản trị: {exc}")
        return

    rem_df_res, has_rem = get_removed_rows()
    rem_df = rem_df_res if (has_rem and isinstance(rem_df_res, pd.DataFrame)) else (rem_df_res[0] if isinstance(rem_df_res, tuple) else pd.DataFrame())
    feat_df, has_feat = get_features_data()
    pred_df, has_pred, loaded_files = get_all_predictions()

    render_section_header(
        overline="04 / MLOPS & DATA GOVERNANCE",
        title="Quản trị Hệ thống, Kiểm toán Mô hình & Toàn vẹn Dữ liệu",
        description=f"Giám sát dữ liệu lịch sử, bộ 29 đặc trưng kỹ thuật, {len(SOURCES)} tổ hợp dự báo Test và ranh giới thời gian."
    )

    methods = tuple(dict.fromkeys(name for name, _, _, _ in SOURCES))
    descriptions = {
        "Naive": ("Persistence (ŷ = P_t), không cần artifact", "Nhóm Tuyến tính & Cơ sở"),
        "Ridge": ("Pipeline hồi quy L2", "Nhóm Tuyến tính & Cơ sở"),
        "Random Forest": ("Tổ hợp cây Random Forest", "Nhóm Mô hình Cây (Bagging)"),
        "XGBoost": ("Gradient Boosting XGBoost", "Nhóm Tăng cường (Boosting)"),
    }
    model_rows = []
    ready_count = 0
    expected_artifacts = (len(methods) - 1) * len(HORIZONS)
    for m_name in methods:
        desc, group_label = descriptions[m_name]
        for h in HORIZONS:
            if m_name == "Naive":
                size_kb = "Trực tiếp"
                validation_status = "Quy tắc trực tiếp"
                operating_status = "Sẵn sàng" if has_clean and not clean_df.empty else "Thiếu dữ liệu Brent"
            else:
                candidates = MODEL_FILE_CANDIDATES.get((m_name, h), [])
                p = candidates[0] if candidates else None
                exists = p is not None and p.is_file()
                size_kb = f"{p.stat().st_size / 1024:.1f} KB" if exists else "N/A"
                can_load = load_trained_model(m_name, h)[2] if exists else False
                validation_status = "ĐẠT" if exists and can_load else ("Lỗi nạp artifact" if exists else "Thiếu artifact")
                operating_status = (
                    "Thiếu dữ liệu Brent" if can_load and (not has_clean or clean_df.empty)
                    else "Sẵn sàng" if can_load else validation_status
                )
                ready_count += int(exists and can_load)
            model_rows.append({
                "Mô hình": m_name,
                "Kỳ hạn": f"t+{h}",
                "Phân nhóm thuật toán": group_label,
                "Kiến trúc & Cấu hình huấn luyện": desc,
                "Kích thước bộ nhớ": size_kb,
                "Trạng thái kiểm định": validation_status,
                "Trạng thái vận hành": operating_status,
            })

    test_data = pred_df.loc[pred_df["split"].eq("test")] if has_pred and "split" in pred_df else pd.DataFrame()
    n_test_rows = len(test_data)
    n_combinations = test_data[["model", "horizon"]].drop_duplicates().shape[0] if n_test_rows else 0
    n_target_days = test_data["target_date"].nunique() if n_test_rows else 0
    n_rem = str(len(rem_df)) if has_rem and isinstance(rem_df, pd.DataFrame) else "N/A"

    split_counts = {}
    training_counts = {}
    purge_counts = {}
    for h in HORIZONS:
        path = FEATURES_DATA_PATH if h in (1, 5) else PROJECT_ROOT / "data" / "processed" / f"brent_features_h{h}.csv"
        split_col, target_col = f"split_{h}", f"target_date_{h}"
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
            if not {"observation_date", split_col, target_col}.issubset(frame.columns):
                continue
            counts = frame[split_col].value_counts().to_dict()
            if not {"train", "val", "test"}.issubset(counts):
                continue
            split_counts[h] = counts
            metadata_path = PROJECT_ROOT / "tv5" / "models" / f"rf_model_t{h}_metadata.json"
            if not metadata_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            purge = metadata["purged_boundary_samples"]
            train = frame.loc[frame[split_col].eq("train")]
            validation = frame.loc[frame[split_col].eq("val")]
            test = frame.loc[frame[split_col].eq("test")]
            train_purge = int((pd.to_datetime(train[target_col]) > pd.to_datetime(validation["observation_date"]).min()).sum())
            validation_purge = int((pd.to_datetime(validation[target_col]) > pd.to_datetime(test["observation_date"]).min()).sum())
            final_rows = len(train) + len(validation) - validation_purge
            if (
                metadata["train_rows"] == len(train)
                and metadata["validation_rows"] == len(validation)
                and metadata["test_rows"] == len(test)
                and purge["train_val"] == train_purge
                and purge["val_test"] == validation_purge
                and metadata["train_rows_used_for_tuning"] == len(train) - train_purge
                and metadata["trainval_rows_used_for_final_fit"] == final_rows
            ):
                training_counts[h] = final_rows
                purge_counts[h] = (train_purge, validation_purge)
        except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError):
            continue

    purge_ready = len(purge_counts) == len(HORIZONS)

    render_bento_kpis([
        {
            "overline": "Dữ liệu sạch & Đặc trưng",
            "badge": "ĐẠT" if (has_clean and has_feat) else "Chờ dữ liệu",
            "value": f"{len(clean_df):,} / {len(feat_df):,} phiên",
            "meta": f"Đã loại {n_rem} phiên thiếu giá",
            "tone": "success" if (has_clean and has_feat) else "warning",
        },
        {
            "overline": "Kho Mô hình Huấn luyện",
            "badge": "Sẵn sàng" if ready_count == expected_artifacts else f"{ready_count}/{expected_artifacts} artifact",
            "value": f"{ready_count} / {expected_artifacts} artifact AI",
            "meta": f"Naive trực tiếp · {len(methods) - 1} mô hình AI × {len(HORIZONS)} kỳ hạn",
            "tone": "success" if ready_count == expected_artifacts else "warning",
        },
        {
            "overline": "Dữ liệu Kiểm định Độc lập",
            "badge": f"{len(loaded_files)}/{len(SOURCES)} tập kết quả",
            "value": f"{n_test_rows:,} bản ghi Test",
            "meta": f"{len(methods)} phương pháp × {len(HORIZONS)} kỳ hạn · {n_combinations} tổ hợp · {n_target_days} ngày mục tiêu",
            "tone": "success" if has_pred and len(loaded_files) == len(SOURCES) and n_combinations == len(SOURCES) else "warning",
        },
        {
            "overline": "Kiểm soát Chống rò rỉ",
            "badge": "ĐẠT" if purge_ready else "Thiếu metadata",
            "value": "Purge theo kỳ hạn" if purge_ready else "Chưa xác minh purge",
            "meta": "Phân tập theo ngày mục tiêu · Cắt biên ranh giới",
            "tone": "success" if purge_ready else "warning",
        },
    ])

    with st.container(border=True):
        st.markdown("#### 1. Bảng kiểm toán trạng thái vận hành các mô hình dự báo")
        st.caption(f"Kiểm kê {len(methods)} phương pháp, {len(HORIZONS)} kỳ hạn theo phiên Brent hợp lệ; dự báo mới dựa trên quan sát cuối trong repository.")
        render_genesis_table(pd.DataFrame(model_rows))

    c_left, c_right = st.columns([1.15, 1])
    with c_left:
        with st.container(border=True):
            st.markdown("#### 2. Kiểm toán tính đồng bộ trên tập Kiểm tra độc lập (Test)")
            st.caption(f"Mỗi tổ hợp cần {EXPECTED_TEST_ROWS} bản ghi Test cùng ngày mục tiêu; dự báo Validation chưa đầy đủ cho bốn phương pháp.")
            bt_audit = []
            for m_name, h, rel_path, _ in SOURCES:
                p = PROJECT_ROOT / rel_path
                total_rows, test_rows, period, status = 0, 0, "N/A", "Thiếu CSV"
                if p.is_file():
                    try:
                        f_df = pd.read_csv(p)
                        total_rows = len(f_df)
                        if list(f_df.columns) != list(COLUMNS):
                            status = "Sai schema"
                        else:
                            t_df = f_df.loc[f_df["split"].eq("test")]
                            test_rows = len(t_df)
                            if test_rows:
                                dates = pd.to_datetime(t_df["target_date"], errors="raise")
                                period = f"{dates.min():%d/%m/%Y} – {dates.max():%d/%m/%Y}"
                            status = (
                                "ĐẠT" if has_pred and test_rows == EXPECTED_TEST_ROWS
                                and test_rows == len(test_data.loc[test_data["model"].eq(m_name) & test_data["horizon"].eq(h)])
                                else "Chưa xác minh"
                            )
                    except (OSError, ValueError, KeyError, pd.errors.ParserError):
                        status = "Không đọc được"
                bt_audit.append({
                    "Mô hình": m_name,
                    "Kỳ hạn": f"t+{h}",
                    "Giai đoạn đánh giá": period,
                    "Tổng bản ghi lưu trữ": total_rows,
                    "Số phiên Test chuẩn": test_rows,
                    "Trạng thái": status,
                })
            render_genesis_table(pd.DataFrame(bt_audit))

    with c_right:
        with st.container(border=True):
            st.markdown("#### 3. Kiểm toán Phân tập & Ranh giới Thời gian")
            st.caption("Train/Validation/Test từ dữ liệu TV3; số final fit Random Forest sau purge từ metadata đã đối chiếu ngày mục tiêu.")
            stages = (
                ("Train theo ngày mục tiêu (trước purge)", lambda h: split_counts.get(h, {}).get("train")),
                ("Validation theo ngày mục tiêu (trước purge)", lambda h: split_counts.get(h, {}).get("val")),
                ("Final fit Random Forest (Train + Validation sau purge)", lambda h: training_counts.get(h)),
                ("Kiểm tra độc lập — Test", lambda h: split_counts.get(h, {}).get("test")),
            )
            split_summary = pd.DataFrame([
                {
                    "Giai đoạn": label,
                    **{f"Kỳ hạn t+{h}": f"{value(h):,} mẫu" if value(h) is not None else "N/A" for h in HORIZONS},
                    "Trạng thái": "ĐẠT" if all(value(h) is not None for h in HORIZONS) else "Thiếu nguồn",
                }
                for label, value in stages
            ])
            render_genesis_table(split_summary)
            render_editorial_note(
                "Quy tắc cắt biên chống rò rỉ (Boundary Purging)",
                "Purge theo ngày mục tiêu tại ranh giới Train/Validation và Validation/Test: "
                + ("; ".join(f"t+{h}: {purge_counts[h][0]}/{purge_counts[h][1]} mẫu" for h in HORIZONS)
                   if purge_ready else "thiếu dữ liệu hoặc metadata để xác minh đủ năm kỳ hạn.")
            )
