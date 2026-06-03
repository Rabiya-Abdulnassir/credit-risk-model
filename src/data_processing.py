import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

from sklearn.utils.validation import check_is_fitted


# =====================================================
# AGGREGATE FEATURES
# =====================================================
class AggregateFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        if "CustomerId" in df.columns and "Amount" in df.columns:

            agg = (
                df.groupby("CustomerId")["Amount"]
                .agg(
                    TotalTransactionAmount="sum",
                    AverageTransactionAmount="mean",
                    TransactionCount="count",
                    StdTransactionAmount="std"
                )
                .reset_index()
            )

            df = df.merge(
                agg,
                on="CustomerId",
                how="left"
            )

            df["StdTransactionAmount"] = (
                df["StdTransactionAmount"]
                .fillna(0)
            )

        return df


# =====================================================
# DATE FEATURES
# =====================================================
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

            df["TransactionHour"] = (
                df["TransactionStartTime"].dt.hour
            )

            df["TransactionDay"] = (
                df["TransactionStartTime"].dt.day
            )

            df["TransactionMonth"] = (
                df["TransactionStartTime"].dt.month
            )

            df["TransactionYear"] = (
                df["TransactionStartTime"].dt.year
            )

            df["IsWeekend"] = (
                df["TransactionStartTime"]
                .dt.dayofweek
                .isin([5, 6])
                .astype(int)
            )

        return df


# =====================================================
# DROP COLUMNS
# =====================================================
class DropColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(
            columns=self.columns,
            errors="ignore"
        )


# =====================================================
# RFM TARGET CREATION
# =====================================================
class RFMTargetCreator(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        snapshot_date=None,
        n_clusters=3,
        random_state=42
    ):
        self.snapshot_date = snapshot_date
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = X.copy()

        df["TransactionStartTime"] = pd.to_datetime(
            df["TransactionStartTime"],
            errors="coerce"
        )

        snapshot_date = (
            pd.to_datetime(self.snapshot_date)
            if self.snapshot_date
            else df["TransactionStartTime"].max()
        )

        rfm = (
            df.groupby("CustomerId")
            .agg(
                Recency=(
                    "TransactionStartTime",
                    lambda x: (snapshot_date - x.max()).days
                ),
                Frequency=("TransactionId", "count"),
                Monetary=("Amount", "sum")
            )
            .reset_index()
        )

        rfm = rfm.fillna(0)

        scaler = StandardScaler()

        rfm_scaled = scaler.fit_transform(
            rfm[
                ["Recency", "Frequency", "Monetary"]
            ]
        )

        n_clusters = min(self.n_clusters, len(rfm))

# fallback if too small dataset
        if n_clusters < 2:
           rfm["Cluster"] = 0
        else:
           kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
            )
        rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
        rfm["Cluster"] = kmeans.fit_predict(
            rfm_scaled
        )

        cluster_summary = (
            rfm.groupby("Cluster")
            [["Recency", "Frequency", "Monetary"]]
            .mean()
        )

        high_risk_cluster = (
            cluster_summary["Recency"]
            .idxmax()
        )

        rfm["is_high_risk"] = (
            rfm["Cluster"] == high_risk_cluster
        ).astype(int)

        df = df.merge(
            rfm[
                ["CustomerId", "is_high_risk"]
            ],
            on="CustomerId",
            how="left"
        )

        return df


# =====================================================
# FEATURE ENGINEERING PIPELINE
# =====================================================
def build_feature_pipeline():

    return Pipeline(
        steps=[
            ("aggregate", AggregateFeatures()),
            ("date_features", DateFeatures()),
            ("rfm_target", RFMTargetCreator()),
            (
                "drop_columns",
                DropColumns(
                    [
                        "TransactionId",
                        "BatchId",
                        "AccountId",
                        "SubscriptionId",
                        "TransactionStartTime"
                    ]
                )
            )
        ]
    )


# =====================================================
# PREPROCESSOR
# =====================================================
def build_preprocessor(df):

    if "is_high_risk" in df.columns:
        df = df.drop(columns=["is_high_risk"])

    if "FraudResult" in df.columns:
        df = df.drop(columns=["FraudResult"])

    numeric_cols = (
        df.select_dtypes(
            include=["int64", "float64"]
        )
        .columns
        .tolist()
    )

    categorical_cols = (
        df.select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_cols
            ),
            (
                "cat",
                categorical_transformer,
                categorical_cols
            )
        ]
    )


# =====================================================
# WOE TRANSFORMER
# =====================================================
class WoETransformer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        feature_cols,
        target_col,
        n_bins=10
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.n_bins = n_bins

    def fit(self, X, y=None):

        self.iv_ = {}

        eps = 1e-9

        total_good = (
            X[self.target_col] == 0
        ).sum()

        total_bad = (
            X[self.target_col] == 1
        ).sum()

        self.binning_ = {}
        self.woe_maps_ = {}

        for col in self.feature_cols:

            try:

                X_tmp = X[[col, self.target_col]].copy()

                X_tmp["bin"] = pd.qcut(
                    X_tmp[col],
                    q=self.n_bins,
                    duplicates="drop"
                )

                grouped = (
                    X_tmp.groupby("bin")
                    [self.target_col]
                    .agg(
                        total="count",
                        bad="sum"
                    )
                )

                grouped["good"] = (
                    grouped["total"]
                    - grouped["bad"]
                )

                grouped["dist_good"] = (
                    grouped["good"]
                    / (total_good + eps)
                )

                grouped["dist_bad"] = (
                    grouped["bad"]
                    / (total_bad + eps)
                )

                grouped["woe"] = np.log(
                    (
                        grouped["dist_good"]
                        + eps
                    )
                    /
                    (
                        grouped["dist_bad"]
                        + eps
                    )
                )

                grouped["iv"] = (
                    grouped["dist_good"]
                    - grouped["dist_bad"]
                ) * grouped["woe"]

                self.iv_[col] = (
                    grouped["iv"]
                    .sum()
                )

            except Exception:
                self.iv_[col] = 0

        return self

    def transform(self, X):
        check_is_fitted(self, ["iv_"])
        return X


# =====================================================
# IV HELPER FUNCTIONS
# =====================================================
def calculate_iv_scores(
    df,
    feature_cols,
    target_col
):

    transformer = WoETransformer(
        feature_cols=feature_cols,
        target_col=target_col
    )

    transformer.fit(df)

    return transformer.iv_


def export_iv_scores(
    iv_scores,
    output_path="artifacts/iv_scores.csv"
):

    iv_df = pd.DataFrame(
        {
            "feature": list(iv_scores.keys()),
            "iv_score": list(iv_scores.values())
        }
    )

    iv_df = iv_df.sort_values(
        "iv_score",
        ascending=False
    )

    iv_df.to_csv(
        output_path,
        index=False
    )

    return iv_df