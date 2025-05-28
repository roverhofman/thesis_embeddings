import logging
from kedro.pipeline import Pipeline, node
from .nodes import rp_images

def create_pipeline(**kwargs) -> Pipeline:
    """
    Recurrence‐Plot image pipeline.
    """
    window = kwargs["window_size"]
    logging.getLogger(__name__).info("RP pipeline (window=%s)", window)
    return Pipeline(
        [
            node(
                func=rp_images,
                inputs=[
                    f"windows_{window}_train",
                    f"windows_{window}_val",
                    f"windows_{window}_test",
                    "params:window_size",
                ],
                outputs=["rp_train", "rp_val", "rp_test"],
                name="rp_image_node",
            ),
        ]
    )
