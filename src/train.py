"""Chronological model fitting for the Brent forecasting experiment."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def make_models() -> dict[str, object]:
    """Create fixed, reproducible model configurations (no test tuning)."""
    return {
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def fit_models(train_x, train_y) -> dict[str, object]:
    """Fit each model on the supplied training rows."""
    fitted = {}
    for name, model in make_models().items():
        model.fit(train_x, train_y)
        fitted[name] = model
    return fitted
