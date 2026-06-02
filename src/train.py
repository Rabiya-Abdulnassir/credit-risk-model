import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn

from src.data_processing import build_pipeline


# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("data/raw/data.csv")

# =========================
# 2. BUILD PIPELINE + TRANSFORM DATA
# =========================
pipeline_builder = build_pipeline()
pipeline = pipeline_builder(df)

df_processed = pipeline.fit_transform(df)

print("Processed shape:", df_processed.shape)

# =========================
# 3. SPLIT DATA
# =========================
target = "is_high_risk"

X = df_processed.drop(columns=[target])
y = df_processed[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 4. EVALUATION FUNCTION
# =========================
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }

    return metrics


# =========================
# 5. MLflow SETUP
# =========================
mlflow.set_experiment("credit-risk-model")

best_model = None
best_score = 0
best_name = ""


# =========================
# 6. MODEL 1: LOGISTIC REGRESSION
# =========================
with mlflow.start_run(run_name="LogisticRegression"):

    log_model = LogisticRegression(max_iter=1000)

    log_model.fit(X_train, y_train)

    metrics = evaluate_model(log_model, X_test, y_test)

    mlflow.log_params({"model": "LogisticRegression"})

    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(log_model, "logistic_model")

    if metrics["roc_auc"] > best_score:
        best_score = metrics["roc_auc"]
        best_model = log_model
        best_name = "LogisticRegression"


# =========================
# 7. MODEL 2: RANDOM FOREST
# =========================
with mlflow.start_run(run_name="RandomForest"):

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    metrics = evaluate_model(rf_model, X_test, y_test)

    mlflow.log_params({
        "model": "RandomForest",
        "n_estimators": 100,
        "max_depth": 10
    })

    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(rf_model, "random_forest_model")

    if metrics["roc_auc"] > best_score:
        best_score = metrics["roc_auc"]
        best_model = rf_model
        best_name = "RandomForest"


# =========================
# 8. BEST MODEL OUTPUT
# =========================
print("\nBest Model:", best_name)
print("Best ROC-AUC:", best_score)

mlflow.sklearn.log_model(best_model, "best_model")