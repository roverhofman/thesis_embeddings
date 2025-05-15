# nodes.py

import pandas as pd
import umap.umap_ as umap  # require umap-learn package (normal umap not working for some reason)

def _to_df(arr, original, n_components):
    """
    Wrap the UMAP output array into a DataFrame, preserving
    the original index (if any) and naming columns UMAP1, UMAP2, ….
    """
    idx = getattr(original, "index", None)
    cols = [f"UMAP{i+1}" for i in range(n_components)]
    return pd.DataFrame(arr, index=idx, columns=cols)

def umap_embedding(train_df, val_df, test_df):
    """
    Apply UMAP embedding to train / validation / test DataFrames.

    Args:
        train_df (pd.DataFrame): Scaled training data.
        val_df   (pd.DataFrame): Scaled validation data.
        test_df  (pd.DataFrame): Scaled test data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
          UMAP-transformed train, validation, and test.
    """
    umap_model = umap.UMAP()
    train_umap = umap_model.fit_transform(train_df)
    val_umap   = umap_model.transform(val_df)
    test_umap  = umap_model.transform(test_df)

    n_comps    = umap_model.n_components
    train_out  = _to_df(train_umap, train_df, n_comps)
    val_out    = _to_df(val_umap,   val_df,   n_comps)
    test_out   = _to_df(test_umap,  test_df,  n_comps)

    return train_out, val_out, test_out
