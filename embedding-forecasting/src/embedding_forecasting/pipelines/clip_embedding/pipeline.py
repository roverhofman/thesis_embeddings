import logging
from kedro.pipeline import Pipeline, node
from .nodes import embed_clip_list

def create_pipeline(**kwargs) -> Pipeline:
    logging.getLogger(__name__).info("Creating CLIP embedding pipeline")
    return Pipeline(
        [
            node(embed_clip_list, "gaf_train", "gaf_clip_train", name="clip_gaf_train"),
            node(embed_clip_list, "gaf_val",   "gaf_clip_val",   name="clip_gaf_val"),
            node(embed_clip_list, "gaf_test",  "gaf_clip_test",  name="clip_gaf_test"),

            node(embed_clip_list, "mtf_train", "mtf_clip_train", name="clip_mtf_train"),
            node(embed_clip_list, "mtf_val",   "mtf_clip_val",   name="clip_mtf_val"),
            node(embed_clip_list, "mtf_test",  "mtf_clip_test",  name="clip_mtf_test"),

            node(embed_clip_list, "rp_train",  "rp_clip_train",  name="clip_rp_train"),
            node(embed_clip_list, "rp_val",    "rp_clip_val",    name="clip_rp_val"),
            node(embed_clip_list, "rp_test",   "rp_clip_test",   name="clip_rp_test"),
        ]
    )
