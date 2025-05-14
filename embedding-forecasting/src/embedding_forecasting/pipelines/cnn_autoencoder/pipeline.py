import pandas as pd
from typing import Dict
from kedro.pipeline import Pipeline, node

from .nodes import train_cnn_autoencoder, evaluate_cnn_autoencoder, visualize_autoencoder

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_cnn_autoencoder,
                inputs=["windows_90", "params:model_options"],
                outputs="node_W90_E32",
                name="cnn_autoencoder_training_node"
            ),
            node(
                func=evaluate_cnn_autoencoder,
                inputs=["node_W90_E32", "windows_90", "params:model_options"],
                outputs=None
            ),
            node(
                func=visualize_autoencoder,
                inputs=["node_W90_E32", "windows_90", "params:model_options"],
                outputs="autoencoder_plots",
            )
        ]
    )
