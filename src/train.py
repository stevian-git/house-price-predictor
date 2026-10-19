from pathlib import Path
import sys, json
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (make_scorer, mean_absolute_error,
                             mean_squared_error, r2_score)

sys.path.append(str(Path(__file__).resolve().parent))
from data_prep import load_and_clean, FEATURES, TARGET

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CAT = ["Neighborhood"]
NUM = ["GrLivArea", "BedroomAbvGr", "Bathrooms", "GarageCars"]


def dollar_scores(y_true_log, y_pred_log):
    a, p = np.expm1(y_true_log), np.expm1(y_pred_log)
    return {"mae": float(mean_absolute_error(a, p)),
            "rmse": float(np.sqrt(mean_squared_error(a, p))),
            "r2": float(r2_score(a, p))}


def build_pipeline(estimator):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", "passthrough", NUM),
    ])
    return Pipeline([("pre", pre), ("model", estimator)])


def main():
    df = load_and_clean()
    X, y = df[FEATURES], np.log1p(df[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    mae_scorer = make_scorer(
        lambda t, p: mean_absolute_error(np.expm1(t), np.expm1(p)),
        greater_is_better=False)

    baselines = {
        "dummy": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(),
        "rf": RandomForestRegressor(n_estimators=200, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }
    metrics = {"baseline": {}}
    for name, est in baselines.items():
        pipe = build_pipeline(est).fit(X_train, y_train)
        metrics["baseline"][name] = dollar_scores(y_test, pipe.predict(X_test))

    rf_pipe = build_pipeline(RandomForestRegressor(random_state=42))
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.5],
    }
    grid = GridSearchCV(rf_pipe, param_grid, cv=5, scoring=mae_scorer, n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    metrics["tuned_rf"] = {
        "best_params": grid.best_params_,
        "cv_mae": float(-grid.best_score_),
        "test": dollar_scores(y_test, best.predict(X_test)),
    }

    feat_names = best.named_steps["pre"].get_feature_names_out()
    imp = best.named_steps["model"].feature_importances_
    top = np.argsort(imp)[::-1][:10]
    metrics["feature_importance_top10"] = {
        str(feat_names[i]): float(imp[i]) for i in top}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODELS_DIR / "model.joblib")
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    t = metrics["tuned_rf"]
    print("Best params:", t["best_params"])
    print("CV MAE:  $%.0f" % t["cv_mae"])
    print("Test:    MAE $%.0f | RMSE $%.0f | R2 %.3f"
          % (t["test"]["mae"], t["test"]["rmse"], t["test"]["r2"]))
    print("Saved:", MODELS_DIR / "model.joblib")


if __name__ == "__main__":
    main()