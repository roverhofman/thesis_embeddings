import logging
from kedro.pipeline import Pipeline, node
from .nodes import embed_resnet_list

def create_pipeline(**kwargs) -> Pipeline:
    logging.getLogger(__name__).info("Creating ResNet50 embedding pipeline")
    return Pipeline(
        [
            node(embed_resnet_list, "gaf_train", "gaf_resnet_train", name="resnet_gaf_train"),
            node(embed_resnet_list, "gaf_val",   "gaf_resnet_val",   name="resnet_gaf_val"),
            node(embed_resnet_list, "gaf_test",  "gaf_resnet_test",  name="resnet_gaf_test"),

            node(embed_resnet_list, "mtf_train", "mtf_resnet_train", name="resnet_mtf_train"),
            node(embed_resnet_list, "mtf_val",   "mtf_resnet_val",   name="resnet_mtf_val"),
            node(embed_resnet_list, "mtf_test",  "mtf_resnet_test",  name="resnet_mtf_test"),

            node(embed_resnet_list, "rp_train",  "rp_resnet_train",  name="resnet_rp_train"),
            node(embed_resnet_list, "rp_val",    "rp_resnet_val",    name="resnet_rp_val"),
            node(embed_resnet_list, "rp_test",   "rp_resnet_test",   name="resnet_rp_test"),
        ]
    )
