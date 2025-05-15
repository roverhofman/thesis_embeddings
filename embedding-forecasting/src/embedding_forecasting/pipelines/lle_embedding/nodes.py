# nodes.py

import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding

def _to_df(arr, original, lle_parameters):
    """
    Wrap the LLE output array into a DataFrame, preserving the original index
    (if present) and naming columns LL1, LL2, ….
    """
    idx = getattr(original, "index", None)
    n_comps = lle_parameters["n_components"]
    cols = [f"LL{i+1}" for i in range(n_comps)]
    return pd.DataFrame(arr, index=idx, columns=cols)

def lle_embedding(train_sc, valid_sc, test_sc, lle_parameters):
    """
    Apply Locally Linear Embedding to pre-scaled train/validation/test sets.

    Args:
        train_sc (np.ndarray or pd.DataFrame): Scaled training data.
        valid_sc (np.ndarray or pd.DataFrame): Scaled validation data.
        test_sc (np.ndarray or pd.DataFrame): Scaled test data.
        lle_parameters (dict): Keys 'n_neighbors', 'n_components', 'random_state'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            Transformed train, validation, and test DataFrames.
    """
    embedding = LocallyLinearEmbedding(
        n_neighbors=lle_parameters["n_neighbors"],
        n_components=lle_parameters["n_components"],
        random_state=lle_parameters["random_state"]
    )

    train_lle = embedding.fit_transform(train_sc)
    valid_lle = embedding.transform(valid_sc)
    test_lle  = embedding.transform(test_sc)

    train_df = _to_df(train_lle, train_sc, lle_parameters)
    valid_df = _to_df(valid_lle, valid_sc, lle_parameters)
    test_df  = _to_df(test_lle, test_sc, lle_parameters)

    return train_df, valid_df, test_df
