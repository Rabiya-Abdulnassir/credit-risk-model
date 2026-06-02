import pandas as pd
from src.data_processing import build_pipeline

# load data
df = pd.read_csv("data/raw/data.csv")

# build pipeline
pipeline_builder = build_pipeline()
pipeline = pipeline_builder(df)

# transform
df_processed = pipeline.fit_transform(df)

print(df_processed.shape)