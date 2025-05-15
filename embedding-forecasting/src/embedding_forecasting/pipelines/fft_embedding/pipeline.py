# pipeline.py

import logging
from kedro.pipeline import Pipeline, node
from .nodes import fft_embedding

def create_pipeline(**kwargs) -> Pipeline:
    """
    Build a Kedro pipeline for FFT embedding.

    Expects in catalog:
      - windows_{window_size}_train
      - windows_{window_size}_val
      - windows_{window_size}_test

    Emits into the catalog:
      - fft_train
      - fft_val
      - fft_test
    """
    logger = logging.getLogger(__name__)
    logger.info("FFT pipeline kwargs: %s", kwargs)

    train_output = "fft_train"
    val_output   = "fft_val"
    test_output  = "fft_test"

    return Pipeline(
        [
            node(
                func=fft_embedding,
                inputs=[
                    f"windows_{kwargs['window_size']}_train",
                    f"windows_{kwargs['window_size']}_val",
                    f"windows_{kwargs['window_size']}_test",
                ],
                outputs=[train_output, val_output, test_output],
                name="fft_embedding_node",
            ),
        ]
    )
