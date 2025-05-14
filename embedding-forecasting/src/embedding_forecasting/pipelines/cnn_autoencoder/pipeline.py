from kedro.pipeline import Pipeline, node
from .nodes import train_cnn_autoencoder, evaluate_cnn_autoencoder, visualize_autoencoder

def create_pipeline(**kwargs) -> Pipeline:
    """
    Constructs the CNN autoencoder pipeline, dynamically wiring
    the train/val windows and naming the model output to include
    window_size and embed_dim.

    Expects:
      - kwargs['window_size']: int
      - kwargs['model_options']: dict with embed_dim, test_size, random_state, epochs, batch_size
    """
    ws = kwargs["window_size"]
    model_opts = kwargs["model_options"]
    ed = model_opts["embed_dim"]

    # Dataset names produced by the data_processing pipeline
    train_ds = f"windows_{ws}_train"
    val_ds = f"windows_{ws}_val"
    # Model output name embeds window and embedding dims
    model_output = f"node_W{ws}_E{ed}"

    return Pipeline([
        # Training node: uses both train and val windows
        node(
            func=train_cnn_autoencoder,
            inputs=[train_ds, val_ds, "params:model_options"],
            outputs=model_output,
            name="cnn_autoencoder_training_node"
        ),
        # Evaluation node: logs train and val losses
        node(
            func=evaluate_cnn_autoencoder,
            inputs=[model_output, train_ds, val_ds, "params:model_options"],
            outputs=None,
            name="cnn_autoencoder_evaluation_node"
        ),
        # Visualization node: produces plots from val window
        node(
            func=visualize_autoencoder,
            inputs=[model_output, val_ds, "params:model_options"],
            outputs="autoencoder_plots",
            name="cnn_autoencoder_visualization_node"
        ),
    ])
