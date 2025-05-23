import logging
from kedro.pipeline import Pipeline, node
from .nodes import pca_embedding


def create_pipeline(**kwargs) -> Pipeline:
    logger = logging.getLogger(__name__)
    logger.info("Embedding pipeline kwargs: %s", kwargs)

    n_comps = kwargs.get("n_components", 10)
    train_output = f"pca_train"
    val_output = f"pca_val"
    test_output = f"pca_test"

    return Pipeline(
        [
            node(
                func=pca_embedding,
                inputs=[
                    f"windows_{kwargs['window_size']}_train",
                    f"windows_{kwargs['window_size']}_val",
                    f"windows_{kwargs['window_size']}_test",
                    "params:n_components",
                ],
                outputs=[train_output, val_output, test_output],
                name="pca_embedding_node",
            ),
        ]
    )
