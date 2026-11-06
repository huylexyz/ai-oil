import struct
import sys
import types
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import PROJECT_ROOT
from src.data_loader import load_brent_data
from src.features import build_features, FEATURE_COLUMNS
from src.backtest_data import load_backtest, SOURCES
from src.core.config import (
    CLEAN_DATA_PATH,
    REMOVED_ROWS_PATH,
    FEATURES_DATA_PATH,
    MODEL_FILE_CANDIDATES,
)


def _parse_ubjson(data: bytes, offset: int = 0):
    t = chr(data[offset])
    offset += 1
    if t in ("Z", "N"):
        return None, offset
    if t == "T":
        return True, offset
    if t == "F":
        return False, offset
    if t == "i":
        return struct.unpack_from(">b", data, offset)[0], offset + 1
    if t == "U":
        return struct.unpack_from(">B", data, offset)[0], offset + 1
    if t == "I":
        return struct.unpack_from(">h", data, offset)[0], offset + 2
    if t == "l":
        return struct.unpack_from(">i", data, offset)[0], offset + 4
    if t == "L":
        return struct.unpack_from(">q", data, offset)[0], offset + 8
    if t == "d":
        return struct.unpack_from(">f", data, offset)[0], offset + 4
    if t == "D":
        return struct.unpack_from(">d", data, offset)[0], offset + 8
    if t == "C":
        return chr(data[offset]), offset + 1
    if t == "S":
        length, offset = _parse_ubjson(data, offset)
        return data[offset : offset + length].decode("utf-8", errors="replace"), offset + length
    if t == "[":
        container_type = None
        if chr(data[offset]) == "$":
            container_type = chr(data[offset + 1])
            offset += 2
        count = None
        if chr(data[offset]) == "#":
            offset += 1
            count, offset = _parse_ubjson(data, offset)
        if container_type and count is not None:
            fmt_map = {"d": (">f", 4), "D": (">d", 8), "l": (">i", 4), "L": (">q", 8), "i": (">b", 1), "U": (">B", 1), "I": (">h", 2)}
            if container_type in fmt_map:
                fmt_char, sz = fmt_map[container_type]
                return np.frombuffer(data, dtype=fmt_char, count=count, offset=offset), offset + count * sz
        res = []
        if count is not None:
            for _ in range(count):
                val, offset = _parse_ubjson(bytes([ord(container_type)]) + data[offset:], 0) if container_type else _parse_ubjson(data, offset)
                res.append(val)
            return res, offset
        while chr(data[offset]) != "]":
            val, offset = _parse_ubjson(data, offset)
            res.append(val)
        return res, offset + 1
    if t == "{":
        if chr(data[offset]) == "$":
            offset += 2
        count = None
        if chr(data[offset]) == "#":
            offset += 1
            count, offset = _parse_ubjson(data, offset)
        res = {}
        if count is not None:
            for _ in range(count):
                k_len, offset = _parse_ubjson(data, offset)
                k = data[offset : offset + k_len].decode("utf-8", errors="replace")
                offset += k_len
                val, offset = _parse_ubjson(data, offset)
                res[k] = val
            return res, offset
        while chr(data[offset]) != "}":
            k_len, offset = _parse_ubjson(data, offset)
            k = data[offset : offset + k_len].decode("utf-8", errors="replace")
            offset += k_len
            val, offset = _parse_ubjson(data, offset)
            res[k] = val
        return res, offset + 1
    raise ValueError(f"Unknown UBJSON tag {t!r}")


class _PureXGBEstimator:
    """Bộ suy luận cây quyết định XGBoost thuần NumPy đảm bảo tương thích trên mọi máy tính."""
    def __init__(self, ubj_dict: dict):
        learner = ubj_dict["Model"]["learner"] if "Model" in ubj_dict else ubj_dict["learner"]
        bm_param = learner["learner_model_param"]
        bs_raw = bm_param["base_score"]
        self.base_score = float(bs_raw.strip("[]")) if isinstance(bs_raw, str) else float(np.ravel(bs_raw)[0])
        self.trees = learner["gradient_booster"]["model"]["trees"]

    def predict(self, X) -> np.ndarray:
        X_mat = X.to_numpy(dtype=np.float32) if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float32)
        preds = np.full(X_mat.shape[0], self.base_score, dtype=np.float64)
        for tree in self.trees:
            left = tree["left_children"]
            right = tree["right_children"]
            split_idx = tree["split_indices"]
            split_cond = tree["split_conditions"]
            default_left = tree["default_left"]
            for r in range(X_mat.shape[0]):
                node = 0
                while left[node] != -1:
                    v = X_mat[r, split_idx[node]]
                    if np.isnan(v):
                        node = left[node] if default_left[node] else right[node]
                    elif v < split_cond[node]:
                        node = left[node]
                    else:
                        node = right[node]
                preds[r] += float(split_cond[node])
        return preds


def _load_xgb_fallback(path: Path):
    class _Dummy:
        def __init__(self, *a, **kw): pass
        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {"_state": state})

    saved = {k: sys.modules.get(k) for k in ("xgboost", "xgboost.sklearn", "xgboost.core", "xgboost.compat")}
    try:
        for mod_name in saved:
            sys.modules[mod_name] = types.ModuleType(mod_name)
        sys.modules["xgboost.core"].Booster = type("Booster", (_Dummy,), {})
        sys.modules["xgboost.sklearn"].XGBRegressor = type("XGBRegressor", (_Dummy,), {})
        sys.modules["xgboost"].XGBRegressor = sys.modules["xgboost.sklearn"].XGBRegressor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            art = joblib.load(path)
        est_obj = art["model"] if isinstance(art, dict) and "model" in art else art
        raw_handle = est_obj._Booster.__dict__["handle"]
        ubj_dict, _ = _parse_ubjson(bytes(raw_handle), 0)
        return _PureXGBEstimator(ubj_dict)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@st.cache_data(show_spinner=False)
def get_clean_data() -> tuple[pd.DataFrame, bool]:
    if CLEAN_DATA_PATH.is_file():
        df = load_brent_data(CLEAN_DATA_PATH)
        df = df.rename(columns={"DCOILBRENTEU": "price"})
        df["DCOILBRENTEU"] = df["price"]
        return df, True
    return pd.DataFrame(), False


@st.cache_data(show_spinner=False)
def get_removed_rows() -> tuple[pd.DataFrame, bool]:
    if REMOVED_ROWS_PATH.is_file():
        df = pd.read_csv(REMOVED_ROWS_PATH)
        return df, True
    return pd.DataFrame(), False


@st.cache_data(show_spinner=False)
def get_features_data() -> tuple[pd.DataFrame, bool]:
    if FEATURES_DATA_PATH.is_file():
        df = pd.read_csv(FEATURES_DATA_PATH)
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        for h in (1, 5):
            if f"target_date_{h}" in df.columns:
                df[f"target_date_{h}"] = pd.to_datetime(df[f"target_date_{h}"])
        return df.sort_values("observation_date").reset_index(drop=True), True

    clean_df, ok = get_clean_data()
    if ok and not clean_df.empty:
        raw_input = clean_df[["observation_date", "DCOILBRENTEU"]].copy()
        df = build_features(raw_input)
        return df, True
    return pd.DataFrame(), False


@st.cache_data(show_spinner=False)
def get_all_predictions() -> tuple[pd.DataFrame, bool, list[str]]:
    """
    Chỉ nạp đúng 442 phiên Test chính thức cho mỗi mô hình thông qua load_backtest(split='test')
    để cả 4 mô hình (Naive, Ridge, Random Forest, XGBoost) đồng nhất tuyệt đối.
    """
    try:
        test_df = load_backtest(split="test").copy()
    except Exception:
        return pd.DataFrame(), False, []

    test_df["error"] = test_df["predicted"] - test_df["actual"]
    test_df["abs_error"] = test_df["error"].abs()

    loaded_files = [rel_path for _, _, rel_path, _ in SOURCES]
    return test_df, True, loaded_files


@st.cache_resource(show_spinner=False)
def load_trained_model(model_name: str, horizon: int):
    if model_name == "Naive":
        return "NAIVE_RULE", "persistence_rule", True

    candidates = MODEL_FILE_CANDIDATES.get((model_name, int(horizon)), [])
    for path in candidates:
        p = Path(path)
        if p.is_file():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    artifact = joblib.load(p)
                estimator = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact
                return estimator, p.name, True
            except Exception:
                if model_name == "XGBoost":
                    try:
                        est_fallback = _load_xgb_fallback(p)
                        return est_fallback, p.name, True
                    except Exception:
                        continue
    return None, None, False


@st.cache_data(show_spinner=False)
def get_rf_tuning_results(horizon: int) -> pd.DataFrame:
    p = PROJECT_ROOT / "tv5" / "results" / f"rf_tuning_h{int(horizon)}.csv"
    if p.is_file():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_feature_importance_data(model_name: str, horizon: int) -> tuple[pd.DataFrame, str]:
    h = int(horizon)
    if model_name == "XGBoost":
        xgb_imp_path = PROJECT_ROOT / "tv6" / "results" / f"xgb_importance_h{h}.csv"
        if xgb_imp_path.is_file():
            df = pd.read_csv(xgb_imp_path)
            total = df["importance_gain"].sum() or 1.0
            out = pd.DataFrame({
                "Hạng": df["rank"].astype(int),
                "Đặc trưng": df["feature"].astype(str),
                "Điểm gốc": df["importance_gain"].round(6),
                "Tỷ trọng (%)": (df["importance_gain"] / total * 100.0).round(2),
            })
            return out, "Normalized Gain"

    estimator, _, ok = load_trained_model(model_name, h)
    if not ok or estimator is None or estimator == "NAIVE_RULE":
        return pd.DataFrame(), ""

    feats = list(FEATURE_COLUMNS)
    scores = None
    if hasattr(estimator, "feature_importances_"):
        scores = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "named_steps"):
        for step in reversed(list(estimator.named_steps.values())):
            if hasattr(step, "coef_"):
                scores = np.abs(np.asarray(step.coef_, dtype=float)).ravel()
                break
            if hasattr(step, "feature_importances_"):
                scores = np.asarray(step.feature_importances_, dtype=float)
                break

    if scores is None or len(scores) != len(feats):
        return pd.DataFrame(), ""

    total = float(scores.sum()) or 1.0
    out = pd.DataFrame({
        "Đặc trưng": feats,
        "Điểm gốc": np.round(scores, 6),
        "Tỷ trọng (%)": np.round(scores / total * 100.0, 2),
    }).sort_values("Tỷ trọng (%)", ascending=False).reset_index(drop=True)
    out.insert(0, "Hạng", np.arange(1, len(out) + 1))
    label_src = "Gini Importance" if model_name == "Random Forest" else "Trọng số chuẩn hóa L2"
    return out, label_src