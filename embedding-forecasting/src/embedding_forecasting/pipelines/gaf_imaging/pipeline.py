import logging
from kedro.pipeline import Pipeline, node
from .nodes import gaf_images

def create_pipeline(**kwargs) -> Pipeline:
    """
    Pipeline that reads windows_{window_size}_{train,val,test} and
    outputs lists of GAF images named gaf_{train,val,test}.
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating GAF‐image pipeline with kwargs: %s", kwargs)
    return Pipeline(
        [
            node(
                func=gaf_images,
                inputs=[
                    f"windows_{kwargs['window_size']}_train",
                    f"windows_{kwargs['window_size']}_val",
                    f"windows_{kwargs['window_size']}_test",
                    "params:window_size"
                ],
                outputs=["gaf_train", "gaf_val", "gaf_test"],
                name="gaf_image_node",
            ),
        ]
    )
