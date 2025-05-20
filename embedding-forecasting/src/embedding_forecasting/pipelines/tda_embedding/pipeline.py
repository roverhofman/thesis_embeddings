# src/<your_package>/pipelines/tda/pipeline.py
import logging
from kedro.pipeline import Pipeline, node

from .nodes import tda_embedding

def create_pipeline(**kwargs) -> Pipeline:
    logger = logging.getLogger(__name__)
    logger.info("TDA embedding pipeline kwargs: %s", kwargs)

    train_output = "tda_train"
    val_output   = "tda_val"
    test_output  = "tda_test"

    return Pipeline(
        [
            node(
                func=tda_embedding,
                inputs=[
                    f"windows_{kwargs['window_size']}_train",
                    f"windows_{kwargs['window_size']}_val",
                    f"windows_{kwargs['window_size']}_test",
                ],
                outputs=[train_output, val_output, test_output],
                name="tda_embedding_node",
            ),
        ]
    )
