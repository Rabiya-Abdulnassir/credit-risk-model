# =========================
# FEATURE ENGINEERING + PROXY TARGET (TASK 3 + TASK 4)
# =========================

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans


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
# 4. RFM PROXY TARGET (TASK 4)
# =========================
class RFMTargetCreator(BaseEstimator, TransformerMixin):

    def __init__(self, snapshot_date=None, n_clusters=3, random_state=42):
        self.snapshot_date = snapshot_date
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = X.copy()

        if "TransactionStartTime" in df.columns:
            df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce")

        snapshot_date = self.snapshot_date
        if snapshot_date is None:
            snapshot_date = df["TransactionStartTime"].max()
        else:
            snapshot_date = pd.to_datetime(snapshot_date)

        # =========================
        # RFM CALCULATION
        # =========================
        rfm = df.groupby("CustomerId").agg(
            Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum")
        ).reset_index()

        rfm = rfm.fillna(0)

        # =========================
        # SCALING
        # =========================
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

        # =========================
        # CLUSTERING
        # =========================
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )

        rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

        # =========================
        # IDENTIFY HIGH RISK
        # =========================
        cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

        high_risk_cluster = cluster_summary["Recency"].idxmax()

        rfm["is_high_risk"] = (rfm["Cluster"] == high_risk_cluster).astype(int)

        # =========================
        # MERGE BACK
        # =========================
        df = df.merge(
            rfm[["CustomerId", "is_high_risk"]],
            on="CustomerId",
            how="left"
        )

        return df


# =========================
# 5. BUILD PIPELINE
# =========================
def build_pipeline():

    feature_pipeline = Pipeline(steps=[
        ("aggregate", AggregateFeatures()),
        ("date_features", DateFeatures()),
        ("rfm_target", RFMTargetCreator()),   # ✅ TASK 4 ADDED HERE
        ("drop_columns", DropColumns([
            "TransactionId",
            "BatchId",
            "AccountId",
            "SubscriptionId",
            "TransactionStartTime"
        ]))
    ])

    def full_pipeline(X):

        X = feature_pipeline.fit_transform(X)

        if "FraudResult" in X.columns:
            X = X.drop(columns=["FraudResult"])

        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

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

        return Pipeline([
            ("feature_engineering", feature_pipeline),
            ("preprocessor", preprocessor)
        ])

    return full_pipeline