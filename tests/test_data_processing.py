from src.data_processing import AggregateFeatures
import pandas as pd


def test_aggregate_features():
    df = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "Amount": [100, 200, 300]
    })

    transformer = AggregateFeatures()
    result = transformer.fit_transform(df)

    assert "TotalTransactionAmount" in result.columns
    assert result.shape[0] >= 1