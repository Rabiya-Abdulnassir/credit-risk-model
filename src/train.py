import os
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from src.data_processing import (
    build_feature_pipeline,
    build_preprocessor,
    calculate_iv_scores,
    export_iv_scores
)

# =========================
# CONFIG
# =========================
RANDOM_STATE = 42
TARGET = "is_high_risk"

os.makedirs("artifacts/plots", exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/raw/data.csv")
print("Loaded dataset shape:", df.shape)

# =========================
# FEATURE ENGINEERING
# =========================
feature_pipeline = build_feature_pipeline()
df_processed = feature_pipeline.fit_transform(df)

print("Processed shape:", df_processed.shape)
print("Processed type:", type(df_processed))

# =========================
# SPLIT DATA
# =========================
X = df_processed.drop(columns=[TARGET])
y = df_processed[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# =========================
# DROP ID COLUMNS
# =========================
drop_cols = ["CustomerId", "ProductId", "ProviderId"]

X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns], errors="ignore")
X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns], errors="ignore")

# =========================
# PREPROCESSING
# =========================
preprocessor = build_preprocessor(X_train)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =========================
# IV SCORE LOGGING
# =========================
numeric_cols = [
    col for col in df_processed.select_dtypes(include=["int64", "float64"]).columns
    if col != TARGET
]

iv_scores = calculate_iv_scores(df_processed, numeric_cols, TARGET)
export_iv_scores(iv_scores)

print("IV scores exported")

# =========================
# EVALUATION
# =========================
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)
    }

# =========================
# ARTIFACT LOGGING
# =========================
def log_model_artifacts(model, X_test, y_test, name):
    cm_path = f"artifacts/plots/{name}_cm.png"
    roc_path = f"artifacts/plots/{name}_roc.png"

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(cm_path)
    plt.close()

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(roc_path)
    plt.close()

    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(roc_path)
    mlflow.log_artifact("artifacts/iv_scores.csv")

# =========================
# MLFLOW SETUP
# =========================
mlflow.set_experiment("credit-risk-model")

best_model = None
best_score = 0
best_name = ""

# =========================
# LOGISTIC REGRESSION
# =========================
with mlflow.start_run(run_name="LogisticRegression"):

    log_model = LogisticRegression(max_iter=1000)

    grid = GridSearchCV(
        log_model,
        param_grid={
            "C": [0.01, 0.1, 1, 10],
            "solver": ["liblinear"]
        },
        cv=5,
        scoring="roc_auc"
    )

    grid.fit(X_train, y_train)
    best_log = grid.best_estimator_

    metrics = evaluate_model(best_log, X_test, y_test)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(
        sk_model=best_log,
        name="model",
        registered_model_name="CreditRiskModel"
    )

    log_model_artifacts(best_log, X_test, y_test, "logistic")

    if metrics["roc_auc"] > best_score:
        best_score = metrics["roc_auc"]
        best_model = best_log
        best_name = "LogisticRegression"

# =========================
# RANDOM FOREST
# =========================
with mlflow.start_run(run_name="RandomForest"):

    rf = RandomForestClassifier(random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        rf,
        param_distributions={
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 20, None]
        },
        n_iter=10,
        cv=3,
        scoring="roc_auc",
        random_state=RANDOM_STATE
    )

    search.fit(X_train, y_train)
    best_rf = search.best_estimator_

    metrics = evaluate_model(best_rf, X_test, y_test)

    mlflow.log_params(search.best_params_)
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(
        sk_model=best_rf,
        name="model",
        registered_model_name="CreditRiskModel"
    )

    log_model_artifacts(best_rf, X_test, y_test, "random_forest")

    if metrics["roc_auc"] > best_score:
        best_score = metrics["roc_auc"]
        best_model = best_rf
        best_name = "RandomForest"

# =========================
# FINAL RESULT
# =========================
print("\nBest Model:", best_name)
print("Best ROC-AUC:", best_score)