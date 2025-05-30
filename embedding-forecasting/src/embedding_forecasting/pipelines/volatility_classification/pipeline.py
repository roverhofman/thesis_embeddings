from kedro.pipeline import Pipeline, node
from .nodes import classification_node


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a single node pipeline that computes and saves
    classification scores for all embeddings.

    Expects the catalog to define:
      - "train_prices": a DataFrame with price_close
      - "test_prices":  a DataFrame with price_close
    """
    return Pipeline(
        [
            node(
                func=classification_node,
                inputs=["train_prices", "test_prices"],
                outputs="vol_embeddings_scores",
                name="classification_node",
            )
        ]
    )
