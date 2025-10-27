import pandas as pd
import numpy as np
from data_handling import get_sample
import os

def redundancy_correlation_matrix(df):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    print(upper)
    # to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    # df_reduced = df.drop(columns=to_drop)


if __name__ == "__main__":
    print(os.getcwd())
    df = get_sample(directory="data/cicids", fraction=0.0001)
    redundancy_correlation_matrix(df)