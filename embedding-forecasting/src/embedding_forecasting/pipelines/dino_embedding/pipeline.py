import logging
from kedro.pipeline import Pipeline, node
from .nodes import embed_dino_list

def create_pipeline(**kwargs) -> Pipeline:
    logging.getLogger(__name__).info("Creating DINOv2 embedding pipeline")
    return Pipeline(
        [
            node(embed_dino_list, "gaf_train", "gaf_dino_train", name="dino_gaf_train"),
            node(embed_dino_list, "gaf_val",   "gaf_dino_val",   name="dino_gaf_val"),
            node(embed_dino_list, "gaf_test",  "gaf_dino_test",  name="dino_gaf_test"),

            node(embed_dino_list, "mtf_train", "mtf_dino_train", name="dino_mtf_train"),
            node(embed_dino_list, "mtf_val",   "mtf_dino_val",   name="dino_mtf_val"),
            node(embed_dino_list, "mtf_test",  "mtf_dino_test",  name="dino_mtf_test"),

            node(embed_dino_list, "rp_train",  "rp_dino_train",  name="dino_rp_train"),
            node(embed_dino_list, "rp_val",    "rp_dino_val",    name="dino_rp_val"),
            node(embed_dino_list, "rp_test",   "rp_dino_test",   name="dino_rp_test"),
        ]
    )
