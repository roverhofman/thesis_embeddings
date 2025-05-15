# nodes.py

import pandas as pd
import numpy as np
import pywt

def _to_df(arr: np.ndarray, original: pd.DataFrame, wavelet_parameters: dict) -> pd.DataFrame:
    """
    Wrap the DWT output array into a DataFrame, preserving the original index
    (if present) and naming columns WAV1, WAV2, ….
    """
    idx = getattr(original, "index", None)
    n_coeffs = arr.shape[1]
    cols = [f"WAV{i+1}" for i in range(n_coeffs)]
    return pd.DataFrame(arr, index=idx, columns=cols)

def wavelet_embedding(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    wavelet_parameters: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply a single‐level 1D discrete wavelet transform to each row of
    train/validation/test and return the approximation coefficients as DataFrames.

    Args:
        train_df, val_df, test_df: each a pd.DataFrame of shape (n_samples, n_features)
        wavelet_parameters:
          - wavelet: str, name of the wavelet basis (e.g. 'db1')
          - level: int, decomposition level (here we use level=1)

    Returns:
        train_wav_df, val_wav_df, test_wav_df: each a pd.DataFrame
            of shape (n_samples, n_coeffs) with columns WAV1…WAVn.
    """
    wavelet = wavelet_parameters.get("wavelet", "db1")
    level   = wavelet_parameters.get("level", 1)

    def _embed(df: pd.DataFrame) -> np.ndarray:
        coeffs_list = []
        for _, row in df.iterrows():
            # wavedec returns [cA_n, cD_n, cD_{n-1}, …]; we take only the cA (approximation)
            coeffs = pywt.wavedec(row.values, wavelet, level=level)
            coeffs_list.append(coeffs[0])
        return np.vstack(coeffs_list)

    train_arr = _embed(train_df)
    val_arr   = _embed(val_df)
    test_arr  = _embed(test_df)

    return (
        _to_df(train_arr, train_df, wavelet_parameters),
        _to_df(val_arr,   val_df,   wavelet_parameters),
        _to_df(test_arr,  test_df,  wavelet_parameters),
    )
