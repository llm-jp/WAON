from PIL import Image
import numpy as np
import requests
from io import BytesIO


def count_unique_colors(img: Image.Image, resize: int = 64, mode: str = "RGB") -> int:
    img = img.convert(mode).resize((resize, resize))
    arr = np.array(img)
    if arr.ndim == 3:
        pixels = arr.reshape(-1, arr.shape[-1])
    else:
        pixels = arr.flatten()
    unique_colors = np.unique(pixels, axis=0)
    return len(unique_colors)


def count_approx_unique_colors(
    img: Image.Image, resize: int = 64, mode: str = "RGB", quantize_bits: int = 4
) -> int:
    img = img.convert(mode).resize((resize, resize))
    arr = np.array(img)
    if arr.ndim == 3:
        pixels = arr.reshape(-1, arr.shape[-1])
    else:
        pixels = arr.flatten()[:, None]

    # 量子化 (e.g., 4bit = 16階調)
    shift = 8 - quantize_bits
    quantized = (pixels >> shift) << shift

    unique_colors = np.unique(quantized, axis=0)
    return len(unique_colors)


def should_discard(img: Image.Image) -> bool:
    w, h = img.size
    # Size filter
    if w < 150 or h < 150:
        return True
    # Aspect ratio filter
    aspect_ratio = w / h
    if aspect_ratio > 2 or aspect_ratio <= 0.5:
        return True
    # Color filter
    n_colors = count_unique_colors(img)
    if n_colors < 32:
        return True
    return False


if __name__ == "__main__":
    # Load image from URL
    # url = "https://www.imai-kawara-yane.com/wp-content/uploads/2020/08/bn03.jpg"
    # url = "https://iro-color.com/img/hd/box-gray.png"
    url = "https://rfs.r-n-i.jp/2018_05_11_19_20_45/14368143.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # Apply filtering
    discard = should_discard(img)
