from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def pca_embedding(train_windows, val_windows, test_windows, n_comps):
    """
    Apply standard scaling followed by PCA embedding to train, validation, and test windows.

    Args:
        train_windows (np.ndarray or pandas.DataFrame): Training window data.
        val_windows (np.ndarray or pandas.DataFrame): Validation window data.
        test_windows (np.ndarray or pandas.DataFrame): Test window data.
        n_comps (int): Number of PCA components.

    Returns:
        Tuple of transformed arrays: (train_pca, val_pca, test_pca)
    """
    # Standard scaling
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_windows)
    val_scaled = scaler.transform(val_windows)
    test_scaled = scaler.transform(test_windows)

    # PCA embedding
    pca = PCA(n_components=n_comps)
    train_pca = pca.fit_transform(train_scaled)
    val_pca = pca.transform(val_scaled)
    test_pca = pca.transform(test_scaled)

    return train_pca, val_pca, test_pca