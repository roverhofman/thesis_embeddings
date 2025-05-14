import logging
from kedro.pipeline import Pipeline, node

from .nodes import split_train_val_test, create_windows


def create_pipeline(**kwargs) -> Pipeline:
    logger = logging.getLogger(__name__)
    logger.info("Data processing kwargs: %s", kwargs)

    ws = kwargs["window_size"]
    vf = kwargs["validation_fraction"]
    tf = kwargs.get("test_fraction", 0.1)

    train_name = f"windows_{ws}_train"
    val_name = f"windows_{ws}_val"
    test_name = f"windows_{ws}_test"

    return Pipeline(
        [
            node(
                func=split_train_val_test,
                inputs=["closing_prices", "params:validation_fraction", "params:test_fraction"],
                outputs=["train_prices", "val_prices", "test_prices"],
                name="split_train_val_test_node",
            ),
            node(
                func=create_windows,
                inputs=["train_prices", "params:window_size"],
                outputs=train_name,
                name="create_train_windows_node",
            ),
            node(
                func=create_windows,
                inputs=["val_prices", "params:window_size"],
                outputs=val_name,
                name="create_val_windows_node",
            ),
            node(
                func=create_windows,
                inputs=["test_prices", "params:window_size"],
                outputs=test_name,
                name="create_test_windows_node",
            ),
        ]
    )