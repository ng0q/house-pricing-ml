from pathlib import Path

import category_encoders as ce
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocess import load_and_clean_data

TARGET = "amount_in_rupees"
DATA_PATH = Path("data/house_prices.csv")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "model.pkl"
TARGET_ENCODER_PATH = MODEL_DIR / "target_encoding.pkl"

RF_PARAMS = {
    "n_estimators": 180,
    "max_depth": 25,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 0.7509776412241582,
    "max_samples": 0.9562634426883665,
    "n_jobs": -1,
    "random_state": 42,
}


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehotencode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    max_categories=20,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(**RF_PARAMS)),
        ]
    )


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(((y_true - y_pred) ** 2).mean())),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def train(data_path: Path = DATA_PATH) -> tuple[object, Pipeline, dict[str, float]]:
    df = load_and_clean_data(str(data_path), target=TARGET)
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    encoder = ce.TargetEncoder(cols=["location"], smoothing=10)
    X_train = encoder.fit_transform(X_train, y_train)
    X_test = encoder.transform(X_test)

    y_train_log = np.log1p(y_train)

    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train_log)

    y_pred = np.expm1(pipeline.predict(X_test))
    metrics = evaluate(y_test, y_pred)
    return encoder, pipeline, metrics


def save_artifacts(encoder, pipeline, model_dir: Path = MODEL_DIR) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_PATH.name
    encoder_path = model_dir / TARGET_ENCODER_PATH.name
    joblib.dump(pipeline, model_path)
    joblib.dump(encoder, encoder_path)
    return model_path, encoder_path


def main() -> None:
    encoder, pipeline, metrics = train()
    model_path, encoder_path = save_artifacts(encoder, pipeline)
    print("=" * 60)
    print("Random Forest Regressor")
    print("=" * 60)
    print(f"MAE:  {metrics['mae']:,.0f} рупий")
    print(f"RMSE: {metrics['rmse']:,.0f} рупий")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"MAPE: {metrics['mape'] * 100:.2f}%")
    print()
    print(f"Saved model:           {model_path}")
    print(f"Saved target encoding: {encoder_path}")
    print("Report reference: MAE 799,909 | R² 0.9414 | MAPE 8.93%")


if __name__ == "__main__":
    main()
