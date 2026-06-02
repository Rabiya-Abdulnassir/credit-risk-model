# src/data_processing.py

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
# 2. DATE FEATURE EXTRACTION
# =========================

class DateFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        if "TransactionStartTime" in df.columns:
            df["TransactionStartTime"] = pd.to_datetime(
                df["TransactionStartTime"],
                errors="coerce"
            )

            df["TransactionHour"] = df["TransactionStartTime"].dt.hour
            df["TransactionDay"] = df["TransactionStartTime"].dt.day
            df["TransactionMonth"] = df["TransactionStartTime"].dt.month
            df["TransactionYear"] = df["TransactionStartTime"].dt.year

        return df


# =========================
# 3. DROP UNNEEDED COLUMNS
# =========================

class DropColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors="ignore")


# =========================
# 4. BUILD PIPELINE
# =========================

def build_pipeline(df):

    df_features = df.drop(columns=["FraudResult"], errors="ignore")

    numeric_features = df_features.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = df_features.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline(steps=[
        ("aggregate", AggregateFeatures()),
        ("date_features", DateFeatures()),
        ("drop_columns", DropColumns([
            "TransactionId",
            "BatchId",
            "AccountId",
            "SubscriptionId",
            "TransactionStartTime"
        ])),
        ("preprocessor", preprocessor)
    ])

    return pipeline