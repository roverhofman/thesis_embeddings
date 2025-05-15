# pipeline.py

import logging
from kedro.pipeline import Pipeline, node
from .nodes import lle_embedding

def create_pipeline(**kwargs) -> Pipeline:
    """
    Build a Kedro pipeline for LLE embedding.

    Expects in catalog:
      - windows_{window_size}_train
      - windows_{window_size}_val
      - windows_{window_size}_test
      - params: lle_parameters (dict with n_neighbors, n_components, random_state)
    Emits:
      - lle_train, lle_val, lle_test
    """
    logger = logging.getLogger(__name__)
    logger.info("LLE pipeline kwargs: %s", kwargs)

    train_output = "lle_train"
    val_output   = "lle_val"
    test_output  = "lle_test"

    return Pipeline(
        [
            node(
                func=lle_embedding,
                inputs=[
                    f"windows_{kwargs['window_size']}_train",
                    f"windows_{kwargs['window_size']}_val",
                    f"windows_{kwargs['window_size']}_test",
                    "params:lle_parameters",
                ],
                outputs=[train_output, val_output, test_output],
                name="lle_embedding_node",
            ),
        ]
    )
