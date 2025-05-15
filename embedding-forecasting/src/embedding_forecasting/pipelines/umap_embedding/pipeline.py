# pipeline.py

import logging
from kedro.pipeline import Pipeline, node
from .nodes import umap_embedding

def create_pipeline(**kwargs) -> Pipeline:
    """
    Build a Kedro pipeline for UMAP embedding.

    Expects in catalog:
      - windows_{window_size}_train
      - windows_{window_size}_val
      - windows_{window_size}_test

    Emits:
      - umap_train, umap_val, umap_test
    """
    logger = logging.getLogger(__name__)
    logger.info("UMAP pipeline kwargs: %s", kwargs)

    train_output = "umap_train"
    val_output   = "umap_val"
    test_output  = "umap_test"

    return Pipeline(
        [
            node(
                func=umap_embedding,
                inputs=[
                    f"windows_{kwargs['window_size']}_train",
                    f"windows_{kwargs['window_size']}_val",
                    f"windows_{kwargs['window_size']}_test",
                ],
                outputs=[train_output, val_output, test_output],
                name="umap_embedding_node",
            ),
        ]
    )
