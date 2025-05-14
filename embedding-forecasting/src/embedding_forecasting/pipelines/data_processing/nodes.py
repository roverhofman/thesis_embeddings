import pandas as pd
from typing import Tuple


def split_train_val_test(
    df: pd.DataFrame, validation_fraction: float, test_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split `df` into train, validation, and test sets.

    Args:
        df: full DataFrame
        validation_fraction: fraction of data to reserve for validation
        test_fraction: fraction of data to reserve for testing

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    n = len(df)
    # Compute split indices
    test_size = int(n * test_fraction)
    val_size = int(n * validation_fraction)
    train_size = n - val_size - test_size

    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size: train_size + val_size].reset_index(drop=True)
    test_df = df.iloc[train_size + val_size:].reset_index(drop=True)
    return train_df, val_df, test_df


def _create_windows(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    series = df["price_close"].astype("float32")
    windows = []
    for i in range(len(series) - window_size + 1):
        window = series.iloc[i : i + window_size]
        mu = window.mean()
        sigma = window.std() if window.std() >= 1e-8 else 1e-8
        windows.append(((window - mu) / sigma).values)
    cols = [f"t{i}" for i in range(window_size)]
    return pd.DataFrame(windows, columns=cols)


def create_windows(
    df: pd.DataFrame, window_size: int
) -> pd.DataFrame:
    """Wrapper so Kedro node signatures stay clean."""
    return _create_windows(df, window_size)