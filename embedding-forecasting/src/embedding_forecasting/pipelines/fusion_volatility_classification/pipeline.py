from kedro.pipeline import Pipeline, node
from .nodes import fusion_classification_node


def create_pipeline(**kwargs) -> Pipeline:
    """
    Pipeline that computes and saves the top-10 fusion classification scores
    for all 2-way embedding combinations.

    Expects the catalog to define:
      - "train_prices": DataFrame with price_close
      - "test_prices":  DataFrame with price_close
    Outputs:
      - "vol_embeddings_fusion_scores": dict with 'binary' and 'multi'
    """
    return Pipeline([
        node(
            func=fusion_classification_node,
            inputs=["train_prices", "test_prices"],
            outputs="fusion_vol_embeddings_scores",
            name="fusion_classification_node",
        )
    ])