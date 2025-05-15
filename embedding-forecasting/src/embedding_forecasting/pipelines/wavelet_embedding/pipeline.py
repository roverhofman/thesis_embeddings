# pipeline.py

import logging
from kedro.pipeline import Pipeline, node
from .nodes import wavelet_embedding

def create_pipeline(**kwargs) -> Pipeline:
    """
    Build a Kedro pipeline for Wavelet‐DWT embedding.

    Expects in catalog:
      - windows_{window_size}_train
      - windows_{window_size}_val
      - windows_{window_size}_test
      - params:wavelet_parameters

    Emits into the catalog:
      - wavelet_train
      - wavelet_val
      - wavelet_test
    """
    logger = logging.getLogger(__name__)
    logger.info("Wavelet pipeline kwargs: %s", kwargs)

    return Pipeline(
        [
            node(
                func=wavelet_embedding,
                inputs=[
                    f"windows_{kwargs['window_size']}_train",
                    f"windows_{kwargs['window_size']}_val",
                    f"windows_{kwargs['window_size']}_test",
                    "params:wavelet_parameters",
                ],
                outputs=[
                    "wavelet_train",
                    "wavelet_val",
                    "wavelet_test",
                ],
                name="wavelet_embedding_node",
            ),
        ]
    )
