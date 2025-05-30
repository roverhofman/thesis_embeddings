import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Dict, Tuple


def _set_global_seed(seed: int):
    """Seed python, numpy, and TF for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _build_cnn_autoencoder(window_size: int, embed_dim: int) -> models.Model:
    """Builds a 1D CNN autoencoder model for a given window length and embedding size."""
    encoder_input = layers.Input(shape=(window_size, 1))
    x = layers.Conv1D(
        filters=16,
        kernel_size=5,
        padding="same",
        activation="relu"
    )(encoder_input)
    x = layers.Flatten()(x)
    embed = layers.Dense(embed_dim, name="embedding")(x)
    x = layers.Dense(window_size * 16, activation="relu")(embed)
    x = layers.Reshape((window_size, 16))(x)
    decoder_output = layers.Conv1D(
        filters=1,
        kernel_size=5,
        padding="same",
        activation=None
    )(x)
    autoencoder = models.Model(encoder_input, decoder_output)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train_cnn_autoencoder(
    train_window: pd.DataFrame,
    val_window: pd.DataFrame,
    parameters: Dict
) -> models.Model:
    """
    Train the CNN autoencoder using explicit training and validation windows,
    with seeds applied for reproducibility.
    """
    # 1) seed everything
    seed = int(parameters["random_state"])
    _set_global_seed(seed)

    # 2) Prepare data
    X_train = train_window.values.astype("float32")[..., np.newaxis]
    X_val   = val_window.values.astype("float32")[..., np.newaxis]

    # 3) Build & fit model
    model = _build_cnn_autoencoder(
        parameters["window_size"],
        parameters["embed_dim"]
    )
    model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=parameters["epochs"],
        batch_size=parameters["batch_size"],
        verbose=0
    )
    return model


def evaluate_cnn_autoencoder(
    model: models.Model,
    train_window: pd.DataFrame,
    val_window: pd.DataFrame,
    parameters: Dict
) -> None:
    """
    Evaluate the autoencoder on both training and validation windows.
    """
    X_train = train_window.values.astype("float32")[..., np.newaxis]
    X_val   = val_window.values.astype("float32")[..., np.newaxis]

    train_loss = model.evaluate(X_train, X_train, verbose=0)
    val_loss   = model.evaluate(X_val, X_val, verbose=0)

    logger = logging.getLogger(__name__)
    logger.info(f"train_mse: {train_loss:.5f} | val_mse: {val_loss:.5f}")


def visualize_autoencoder(
    model: models.Model,
    val_window: pd.DataFrame,
    parameters: Dict
) -> Dict[str, plt.Figure]:
    """
    Produce reconstruction and embedding PCA plots using the validation window.
    """
    W = parameters["window_size"]
    E = parameters["embed_dim"]

    X = val_window.values.astype("float32")[..., np.newaxis]

    # Reconstruction of first sample
    orig = X[0, :, 0]
    recon = model.predict(X[0:1], batch_size=parameters["batch_size"])[0, :, 0]

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(orig, label="Original", linewidth=2)
    ax1.scatter(
        np.arange(W), recon,
        label="Reconstructed", edgecolor="k", s=50, marker="o"
    )
    ax1.set_title(f"CNN Reconstruction: {W}-day window, {E}-dim embed", fontsize=14)
    ax1.set_xlabel("Day (t)", fontsize=12)
    ax1.set_ylabel("Value", fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=True, shadow=True)
    fig1.tight_layout()

    # PCA of embeddings for visualization
    encoder = models.Model(inputs=model.input,
                           outputs=model.get_layer("embedding").output)
    embeds = encoder.predict(X, batch_size=parameters["batch_size"])
    coords = PCA(n_components=2).fit_transform(embeds)
    net_change = X[:, -1, 0] - X[:, 0, 0]
    up_trend = net_change > 0

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(
        coords[~up_trend, 0], coords[~up_trend, 1],
        label="Down Trend", marker="o", edgecolor="k", s=40
    )
    ax2.scatter(
        coords[ up_trend, 0], coords[ up_trend, 1],
        label="Up Trend",   marker="^", edgecolor="k", s=40
    )
    ax2.set_title(f"PCA of {W}-day Embeddings", fontsize=14)
    ax2.set_xlabel("PC1", fontsize=12)
    ax2.set_ylabel("PC2", fontsize=12)
    ax2.grid(alpha=0.2)
    ax2.legend(frameon=True, shadow=True)
    fig2.tight_layout()

    return {
        "reconstruction.png": fig1,
        "pca.png": fig2
    }


def save_cnn_embeddings(
    model: models.Model,
    train_window: pd.DataFrame,
    val_window: pd.DataFrame,
    test_window: pd.DataFrame,
    parameters: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract and save embeddings for train, validation, and test windows.
    """
    # Prepare data
    X_train = train_window.values.astype("float32")[..., np.newaxis]
    X_val   = val_window.values.astype("float32")[..., np.newaxis]
    X_test  = test_window.values.astype("float32")[..., np.newaxis]

    # Build encoder
    encoder = models.Model(inputs=model.input,
                           outputs=model.get_layer("embedding").output)

    # Predict embeddings
    emb_train = encoder.predict(X_train, batch_size=parameters["batch_size"])
    emb_val   = encoder.predict(X_val,   batch_size=parameters["batch_size"])
    emb_test  = encoder.predict(X_test,  batch_size=parameters["batch_size"])

    # Wrap into DataFrames
    idx_train = getattr(train_window, "index", None)
    idx_val   = getattr(val_window,   "index", None)
    idx_test  = getattr(test_window,  "index", None)
    cols = [f"EMB{i+1}" for i in range(parameters["embed_dim"])]

    df_train = pd.DataFrame(emb_train, index=idx_train, columns=cols)
    df_val   = pd.DataFrame(emb_val,   index=idx_val,   columns=cols)
    df_test  = pd.DataFrame(emb_test,  index=idx_test,  columns=cols)

    return df_train, df_val, df_test