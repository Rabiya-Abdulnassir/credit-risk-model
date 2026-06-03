import pandas as pd

from src.data_processing import (
    AggregateFeatures,
    DateFeatures,
    DropColumns,
    RFMTargetCreator
)

# =========================
# TEST 1: AGGREGATE FEATURES
# =========================
def test_aggregate_features():
    df = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "Amount": [100, 200, 300]
    })

    transformer = AggregateFeatures()
    result = transformer.fit_transform(df)

    # check columns exist
    assert "TotalTransactionAmount" in result.columns
    assert "AverageTransactionAmount" in result.columns

    # check aggregation correctness
    cust1_total = result[result["CustomerId"] == 1]["TotalTransactionAmount"].iloc[0]
    assert cust1_total == 300


# =========================
# TEST 2: DATE FEATURES
# =========================
def test_date_features():
    df = pd.DataFrame({
        "TransactionStartTime": ["2024-01-01 10:00:00"]
    })

    transformer = DateFeatures()
    result = transformer.fit_transform(df)

    assert "TransactionHour" in result.columns
    assert result["TransactionHour"].iloc[0] == 10


# =========================
# TEST 3: DROP COLUMNS
# =========================
def test_drop_columns():
    df = pd.DataFrame({
        "A": [1, 2],
        "B": [3, 4]
    })

    transformer = DropColumns(columns=["B"])
    result = transformer.fit_transform(df)

    assert "B" not in result.columns
    assert "A" in result.columns


# =========================
# TEST 4: RFM TARGET CREATION (BASIC CHECK)
# =========================
def test_rfm_target_creation():
    df = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "TransactionId": [10, 11, 20],
        "Amount": [100, 200, 300],
        "TransactionStartTime": [
            "2024-01-01 10:00:00",
            "2024-01-02 10:00:00",
            "2024-01-01 10:00:00"
        ]
    })

    transformer = RFMTargetCreator()
    result = transformer.fit_transform(df)

    assert "is_high_risk" in result.columns
    assert result["is_high_risk"].isin([0, 1]).all()