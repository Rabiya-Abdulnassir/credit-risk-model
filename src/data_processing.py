# =========================
# TASK 3 - FEATURE ENGINEERING PIPELINE (CLEAN VERSION)
# =========================

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =========================
# 1. AGGREGATE FEATURES
# =========================
class AggregateFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        if "CustomerId" in df.columns and "Amount" in df.columns:
            agg = df.groupby("CustomerId")["Amount"].agg(
                TotalTransactionAmount="sum",
                AverageTransactionAmount="mean",
                TransactionCount="count",
                StdTransactionAmount="std"
            ).reset_index()

            df = df.merge(agg, on="CustomerId", how="left")

        return df


# =========================
# 2. DATE FEATURES
# =========================
class DateFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        if "TransactionStartTime" in df.columns:
            df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce")

            df["TransactionHour"] = df["TransactionStartTime"].dt.hour
            df["TransactionDay"] = df["TransactionStartTime"].dt.day
            df["TransactionMonth"] = df["TransactionStartTime"].dt.month
            df["TransactionYear"] = df["TransactionStartTime"].dt.year

        return df


# =========================
# 3. DROP COLUMNS
# =========================
class DropColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors="ignore")


# =========================
# 4. BUILD TASK 3 PIPELINE
# =========================
def build_pipeline():

    feature_pipeline = Pipeline(steps=[
        ("aggregate", AggregateFeatures()),
        ("date_features", DateFeatures()),
        ("drop_columns", DropColumns([
            "TransactionId",
            "BatchId",
            "AccountId",
            "SubscriptionId",
            "TransactionStartTime"
        ]))
    ])

    def full_pipeline(X):

        # STEP 1: feature engineering FIRST
        X = feature_pipeline.fit_transform(X)

        # STEP 2: drop target if exists
        if "FraudResult" in X.columns:
            X = X.drop(columns=["FraudResult"])

        # STEP 3: SAFE column detection AFTER transformation
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # STEP 4: preprocessors
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ])

        # STEP 5: FINAL PIPELINE
        final_pipeline = Pipeline([
            ("feature_engineering", feature_pipeline),
            ("preprocessor", preprocessor)
        ])

        return final_pipeline

    return full_pipeline