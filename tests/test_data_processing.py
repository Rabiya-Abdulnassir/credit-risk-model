import pandas as pd
from src.data_processing import build_pipeline


def test_pipeline_returns_dataframe():
    df = pd.DataFrame({
        "CustomerId": ["A", "A", "B"],
        "Amount": [100, 200, 300],
        "TransactionId": [1, 2, 3],
        "TransactionStartTime": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "BatchId": [10, 11, 12],
        "AccountId": [100, 100, 101],
        "SubscriptionId": [1, 1, 2]
    })

    pipeline = build_pipeline(df)
    output = pipeline.fit_transform(df)

    assert output is not None


def test_target_column_exists():
    df = pd.DataFrame({
        "CustomerId": ["A", "A"],
        "Amount": [100, 200],
        "TransactionId": [1, 2],
        "TransactionStartTime": ["2023-01-01", "2023-01-02"],
        "BatchId": [10, 11],
        "AccountId": [100, 100],
        "SubscriptionId": [1, 1]
    })

    pipeline = build_pipeline(df)
    output = pipeline.fit_transform(df)

    assert "is_high_risk" in output.columns