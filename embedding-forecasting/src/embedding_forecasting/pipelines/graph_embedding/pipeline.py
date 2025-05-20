import logging
from kedro.pipeline import Pipeline, node
from .nodes import graph_embedding


def create_pipeline(**kwargs) -> Pipeline:
    logger = logging.getLogger(__name__)
    logger.info("Graph embedding pipeline kwargs: %s", kwargs)

    train_output = "graph_train"
    val_output = "graph_val"
    test_output = "graph_test"

    return Pipeline(
        [
            node(
                func=graph_embedding,
                inputs=[
                    f"windows_{kwargs['window_size']}_train",
                    f"windows_{kwargs['window_size']}_val",
                    f"windows_{kwargs['window_size']}_test",
                ],
                outputs=[train_output, val_output, test_output],
                name="graph_embedding_node",
            ),
        ]
    )
