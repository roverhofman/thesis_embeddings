import numpy as np
import pandas as pd
from pyts.image import MarkovTransitionField
from PIL import Image

def ts_to_mtf(ts: np.ndarray, image_size: int | None = None) -> np.ndarray:
    """
    Compute the raw MTF transform array for a 1D time series.
    """
    ts = np.array(ts).reshape(1, -1)
    mtf = MarkovTransitionField(image_size=image_size)
    return mtf.transform(ts)[0]

def _normalize_and_convert(arr: np.ndarray) -> Image.Image:
    """
    Normalize a 2D array to [0,255] uint8 and convert to RGB PIL Image.
    """
    img_min, img_max = arr.min(), arr.max()
    if img_max > img_min:
        norm = (arr - img_min) / (img_max - img_min)
    else:
        norm = np.zeros_like(arr)
    uint8 = (norm * 255).astype(np.uint8)
    return Image.fromarray(uint8).convert("RGB")

def mtf_images(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    image_size: int | None = None
) -> tuple[list[Image.Image], list[Image.Image], list[Image.Image]]:
    """
    Convert each row of train/val/test DataFrames into normalized RGB MTF images.
    Returns three lists of PIL.Image objects of size (image_size, image_size).
    """
    raw_train = [ts_to_mtf(row, image_size) for row in train_df.values]
    raw_valid = [ts_to_mtf(row, image_size) for row in valid_df.values]
    raw_test  = [ts_to_mtf(row, image_size) for row in test_df.values]

    train_imgs = [_normalize_and_convert(arr) for arr in raw_train]
    valid_imgs = [_normalize_and_convert(arr) for arr in raw_valid]
    test_imgs  = [_normalize_and_convert(arr) for arr in raw_test]

    return train_imgs, valid_imgs, test_imgs
