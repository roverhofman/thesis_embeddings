import numpy as np
import pandas as pd
from pyts.image import GramianAngularField
from PIL import Image


def ts_to_gaf(ts: np.ndarray, image_size: int | None = None) -> np.ndarray:
    """
    Compute the raw GAF transform array for a 1D time series.
    """
    ts = ts.reshape(1, -1)
    gaf = GramianAngularField(image_size=image_size, method='summation')
    return gaf.transform(ts)[0]


def _normalize_and_convert(img: np.ndarray) -> Image.Image:
    """
    Normalize a 2D array to [0,255] uint8 and convert to RGB PIL Image.
    """
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img_norm = (img - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img)
    img_uint8 = (img_norm * 255).astype(np.uint8)
    return Image.fromarray(img_uint8).convert("RGB")


def gaf_images(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    image_size: int | None = None
) -> tuple[list[Image.Image], list[Image.Image], list[Image.Image]]:
    """
    Convert each row of train/val/test DataFrames into normalized RGB GAF images.
    Returns three lists of PIL.Image objects of size (image_size, image_size).
    """
    # Raw GAF arrays
    raw_train = [ts_to_gaf(row, image_size) for row in train_df.values]
    raw_valid = [ts_to_gaf(row, image_size) for row in valid_df.values]
    raw_test  = [ts_to_gaf(row, image_size) for row in test_df.values]

    # Normalize and convert
    train_imgs = [_normalize_and_convert(arr) for arr in raw_train]
    valid_imgs = [_normalize_and_convert(arr) for arr in raw_valid]
    test_imgs  = [_normalize_and_convert(arr) for arr in raw_test]

    return train_imgs, valid_imgs, test_imgs
