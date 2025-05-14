"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import create_multiple_windows

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_multiple_windows,
            inputs=["closing_prices", "params:window_sizes"],
            outputs=[
                *[f"windows_{size}" for size in [30, 60, 90, 120]]
            ],
            name="create_multiple_windows_node"
            ),
        ])
