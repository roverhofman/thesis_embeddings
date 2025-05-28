import logging
from kedro.pipeline import Pipeline, node
from .nodes import mtf_images

def create_pipeline(**kwargs) -> Pipeline:
    """
    MTF image pipeline.
    """
    window = kwargs["window_size"]
    logging.getLogger(__name__).info("MTF pipeline (window=%s)", window)
    return Pipeline(
        [
            node(
                func=mtf_images,
                inputs=[
                    f"windows_{window}_train",
                    f"windows_{window}_val",
                    f"windows_{window}_test",
                    "params:window_size",
                ],
                outputs=["mtf_train", "mtf_val", "mtf_test"],
                name="mtf_image_node",
            ),
        ]
    )
