import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans


# =========================
# 1. RFM TARGET CREATION
# =========================

class RFMTargetCreator(BaseEstimator, TransformerMixin):

    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

        snapshot_date = self.snapshot_date or df["TransactionStartTime"].max()

        rfm = df.groupby("CustomerId").agg(
            Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum")
        ).reset_index()

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

        cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

        high_risk_cluster = cluster_summary["Recency"].idxmax()

        rfm["is_high_risk"] = (rfm["Cluster"] == high_risk_cluster).astype(int)

        df = df.merge(rfm[["CustomerId", "is_high_risk"]], on="CustomerId", how="left")

        return df


# =========================
# 2. AGGREGATE FEATURES
# =========================

class AggregateFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        agg = df.groupby("CustomerId")["Amount"].agg(
            TotalTransactionAmount="sum",
            AverageTransactionAmount="mean",
            TransactionCount="count",
            StdTransactionAmount="std"
        ).reset_index()

        return df.merge(agg, on="CustomerId", how="left")


# =========================
# 3. DATE FEATURES
# =========================

class DateFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

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
# 4. DROP COLUMNS
# =========================

class DropColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors="ignore")


# =========================
# 5. BUILD PIPELINE
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
        ("rfm_target", RFMTargetCreator()),
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