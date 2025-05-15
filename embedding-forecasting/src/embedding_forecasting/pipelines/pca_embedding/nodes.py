import pandas as pd
from sklearn.decomposition import PCA


def _to_df(arr, original, n_comps):
    idx = getattr(original, "index", None)
    cols = [f"PC{i+1}" for i in range(n_comps)]
    return pd.DataFrame(arr, index=idx, columns=cols)


def pca_embedding(train_windows, val_windows, test_windows, n_comps):
    """
    Apply PCA embedding to pre-scaled train, validation, and test windows,
    returning pandas DataFrames.

    Args:
        train_windows (np.ndarray or pandas.DataFrame): Scaled training window data.
        val_windows (np.ndarray or pandas.DataFrame): Scaled validation window data.
        test_windows (np.ndarray or pandas.DataFrame): Scaled test window data.
        n_comps (int): Number of PCA components.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            PCA-transformed train, validation, and test DataFrames.
    """
    # PCA embedding
    pca = PCA(n_components=n_comps)
    train_pca = pca.fit_transform(train_windows)
    val_pca = pca.transform(val_windows)
    test_pca = pca.transform(test_windows)

    # Helper to wrap into DataFrame and preserve index if present

    train_df = _to_df(train_pca, train_windows, n_comps)
    val_df = _to_df(val_pca, val_windows, n_comps)
    test_df = _to_df(test_pca, test_windows, n_comps)

    return train_df, val_df, test_df
