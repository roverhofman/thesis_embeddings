# nodes.py

import pandas as pd
import numpy as np
from scipy.fft import fft

def _to_df(arr: np.ndarray, original: pd.DataFrame) -> pd.DataFrame:
    """
    Wrap the FFT output array into a DataFrame, preserving the original index
    (if present) and naming columns FFT1, FFT2, ….
    """
    idx = getattr(original, "index", None)
    n_cols = arr.shape[1]
    cols = [f"FFT{i+1}" for i in range(n_cols)]
    return pd.DataFrame(arr, index=idx, columns=cols)

def fft_embedding(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply row‐wise Fast Fourier Transform (magnitude) to train/validation/test.

    Args:
        train_df, valid_df, test_df: each a pd.DataFrame of shape (n_samples, n_features)

    Returns:
        train_fft_df, valid_fft_df, test_fft_df: each a pd.DataFrame
            of shape (n_samples, n_features), columns FFT1…FFTn.
    """
    # compute magnitude of the FFT along each row
    train_arr = np.apply_along_axis(lambda x: np.abs(fft(x)), axis=1, arr=train_df.values)
    valid_arr = np.apply_along_axis(lambda x: np.abs(fft(x)), axis=1, arr=valid_df.values)
    test_arr  = np.apply_along_axis(lambda x: np.abs(fft(x)), axis=1, arr=test_df.values)

    return (
        _to_df(train_arr, train_df),
        _to_df(valid_arr, valid_df),
        _to_df(test_arr,  test_df),
    )
