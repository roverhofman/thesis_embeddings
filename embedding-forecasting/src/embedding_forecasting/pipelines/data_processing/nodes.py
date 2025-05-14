"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""
import pandas as pd
from typing import Dict, List


def _create_windows(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    series = df['price_close'].astype('float32')

    windows = []
    for i in range(len(series) - window_size + 1):
        window = series.iloc[i : i + window_size]
        mu = window.mean()
        sigma = window.std()
        if sigma < 1e-8:
            sigma = 1e-8  # avoid division by zero
        # z-score normalization
        normed = (window - mu) / sigma
        windows.append(normed.values)

    # Create column names for each time step
    cols = [f"t{i}" for i in range(window_size)]
    # Construct and return a DataFrame
    return pd.DataFrame(windows, columns=cols)


def create_multiple_windows(df: pd.DataFrame, window_sizes: List[int]) -> List[pd.DataFrame]:
    """
    Generate a list of normalized-window DataFrames for each window_size.
    The position in the returned list matches the position in window_sizes.
    """
    return [_create_windows(df, size) for size in window_sizes]