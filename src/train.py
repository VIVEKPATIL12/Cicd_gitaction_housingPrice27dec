import os
import json
import joblib
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

from utils import load_params, ensure_dirs

def build_model(model_type: str, params: dict):
    if model_type == "LinearRegression":
        return LinearRegression(**params)
    if model_type == "Ridge":
        return Ridge(**params)
    if model_type == "RandomForestRegressor":
        return RandomForestRegressor(**params)
    if model_type == "SVR":
        return SVR(**params)
    if model_type == "KNeighborsRegressor":
        return KNeighborsRegressor(**params)
    if model_type == "XGBRegressor":
        if not XGB_AVAILABLE:
            raise RuntimeError("XGBoost not installed in this environment.")
        return XGBRegressor(**params)
    raise ValueError(f"Unknown model type: {model_type}")

def build_preprocessor(categorical_features, numeric_features):
    cat = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num = Pipeline([
        ("scaler", StandardScaler(with_mean=False))  # sparse safety for OHE downstream
    ])
    pre = ColumnTransformer([
        ("cat", cat, categorical_features),
        ("num", num, numeric_features)
    ], remainder="drop")
    return pre

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def main():
    ensure_dirs()
    params = load_params()

    df = pd.read_csv("outputs/cleaned_housing.csv")
    target = params["target"]
    if target not in df.columns:
        raise RuntimeError(f"Target column '{target}' not found in data.")

    categorical_features = [c for c in params["categorical_features"] if c in df.columns]
    numeric_features = [c for c in params["numeric_features"] if c in df.columns]

    X = df[categorical_features + numeric_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params.get("test_size", 0.2), random_state=params.get("random_state", 42)
    )

    preprocessor = build_preprocessor(categorical_features, numeric_features)

    all_metrics = []
    best_rmse = float("inf")
    best_artifact = None

    for model_key, model_cfg in params["models"].items():
        model_type = model_cfg["type"]
        for i, pset in enumerate(model_cfg["param_sets"]):
            # Build pipeline: preprocessor + model
            reg = build_model(model_type, pset)
            pipe = Pipeline([
                ("preprocess", preprocessor),
                ("model", reg)
            ])

            # Fit + predict
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            # Evaluate
            m = evaluate(y_test, preds)
            run_name = f"{model_key}_set{i+1}"
            metrics_path = f"outputs/metrics/metrics_{run_name}.json"
            model_path = f"outputs/models/model_{run_name}.pkl"

            # Save metrics and model
            with open(metrics_path, "w") as f:
                json.dump({
                    "model_key": model_key,
                    "model_type": model_type,
                    "param_set_index": i+1,
                    "params": pset,
                    "metrics": m
                }, f, indent=4)
            joblib.dump(pipe, model_path)

            print(f"[{run_name}] MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} R2={m['R2']:.3f}")

            # Track best by RMSE
            if m["RMSE"] < best_rmse:
                best_rmse = m["RMSE"]
                best_artifact = {"run_name": run_name, "metrics": m, "model_path": model_path}

            # Collect summary
            all_metrics.append({
                "run_name": run_name,
                "model_type": model_type,
                "params": pset,
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "R2": m["R2"]
            })

    # Save summary
    pd.DataFrame(all_metrics).to_csv("outputs/metrics/all_results.csv", index=False)

    # Save best alias
    if best_artifact:
        best_link_path = "outputs/models/best_model.pkl"
        # Copy by re-dumping
        best_model = joblib.load(best_artifact["model_path"])
        joblib.dump(best_model, best_link_path)
        with open("outputs/metrics/best_metrics.json", "w") as f:
            json.dump({
                "best_run": best_artifact["run_name"],
                "metrics": best_artifact["metrics"]
            }, f, indent=4)

        print(f"Best model: {best_artifact['run_name']} (RMSE={best_artifact['metrics']['RMSE']:.3f})")

if __name__ == "__main__":
    main()
