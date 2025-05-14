"""
This is a boilerplate pipeline 'cnn_autoencoder'
generated using Kedro 0.19.12
"""

import numpy as np
import pandas as pd
import typing as t
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import logging 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def _build_cnn_autoencoder(window_size: int, embed_dim: int) -> models.Model:
    """Builds a 1D CNN autoencoder model for a given window length and embedding size."""
    encoder_input = layers.Input(shape=(window_size, 1))
    # Conv layer (same padding) to extract features
    x = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation='relu')(encoder_input)
    x = layers.Flatten()(x)
    embed = layers.Dense(embed_dim, name='embedding')(x)
    # Decoder
    x = layers.Dense(window_size * 16, activation='relu')(embed)
    x = layers.Reshape((window_size, 16))(x)
    decoder_output = layers.Conv1D(filters=1, kernel_size=5, padding='same', activation=None)(x)
    autoencoder = models.Model(encoder_input, decoder_output)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder



def train_cnn_autoencoder(
    window: pd.DataFrame,
    parameters: t.Dict
	) -> models.Model:

    X = window.values.astype('float32')[..., np.newaxis]
    X_train, X_val = train_test_split(
        X, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    model = _build_cnn_autoencoder(parameters["window_size"], parameters["embed_dim"])
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
    window: pd.DataFrame,
    parameters: t.Dict
	):

    X = window.values.astype('float32')[..., np.newaxis]
    X_train, X_val = train_test_split(
        X, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    train_loss = model.evaluate(X_train, X_train, verbose=0)
    val_loss = model.evaluate(X_val, X_val, verbose=0)

    logger = logging.getLogger(__name__)
    logger.info(f"train_mse: {train_loss:.5f} | val_mse: {val_loss:.5f}")



def visualize_autoencoder(
    model: models.Model,
    window: pd.DataFrame,
    parameters: t.Dict
) -> t.Dict[str, plt.Figure]:

    W = parameters["window_size"]
    E = parameters["embed_dim"]

    # Convert DataFrame to numpy array
    X = window.values.astype('float32')

    split = int(parameters["test_size"] * len(X))
    X_val = X[split:][..., np.newaxis]

    # Reconstruction Plot
    orig = X_val[0, :, 0]
    recon = model.predict(X_val[0:1], batch_size=parameters["batch_size"])[0, :, 0]

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(orig, label="Original", color="gold", linewidth=2)
    ax1.scatter(
        np.arange(W), recon,
        label="Reconstructed", color="darkorange",
        edgecolor="k", s=50, marker="o"
    )
    ax1.set_title(f"CNN Reconstruction: {W}-day window, {E}-dim embed", fontsize=14)
    ax1.set_xlabel("Day (t)", fontsize=12)
    ax1.set_ylabel("Value", fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=True, shadow=True)
    fig1.tight_layout()

    # PCA Plot
    encoder = models.Model(inputs=model.input,
                    outputs=model.get_layer('embedding').output)
    embeds = encoder.predict(X_val, batch_size=parameters["batch_size"])

    coords = PCA(n_components=2).fit_transform(embeds)
    net_change = X_val[:, -1, 0] - X_val[:, 0, 0]
    up_trend = net_change > 0

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(
        coords[~up_trend, 0], coords[~up_trend, 1],
        label="Down Trend", marker="o", edgecolor="k",
        s=40, c="tomato"
    )
    ax2.scatter(
        coords[ up_trend, 0], coords[ up_trend, 1],
        label="Up Trend",   marker="^", edgecolor="k",
        s=40, c="limegreen"
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
