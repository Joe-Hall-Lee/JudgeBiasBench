"""Quick look at an EasyR1 parquet split: python read_parquet.py data/train/generative/grpo/bias_0p25/train-orig.parquet"""
import sys

import pandas as pd

df = pd.read_parquet(sys.argv[1])
df.info()
print(df["answer"].value_counts())
print(df.head())
