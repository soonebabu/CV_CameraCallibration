"""
Classical human boundary segmentation for thermal (infrared) images.

A person is usually the warmest object in a thermal frame, so a global
intensity threshold (Otsu) on the (optionally blurred) grayscale frame
separates the body from the background directly - no learned model
involved. Works on both raw single-channel thermal frames and false-color
palette JPEGs (converted to grayscale luminance first, which tracks
intensity reasonably well for monotonic palettes like ironbow/white-hot).
"""

import cv2
import numpy as np

from .metrics import keep_largest_component


def thermal_segment(img, hot_is_bright=True, blur_ksize=5):
    """
    img: BGR or single-channel thermal frame.
    hot_is_bright: set False for "black-hot" palettes/cameras where the
        warm subject renders darker than the background.
    Returns (binary_mask uint8 {0,1}, largest_contour or None, gray_used).
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if blur_ksize > 1:
        k = blur_ksize + 1 if blur_ksize % 2 == 0 else blur_ksize
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    working = gray if hot_is_bright else (255 - gray)

    _, mask = cv2.threshold(working, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask = (mask > 0).astype(np.uint8)

    kernel = np.ones((7, 7), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    if binary_mask.sum() > 0:
        binary_mask = keep_largest_component(binary_mask)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = max(contours, key=cv2.contourArea) if contours else None

    return binary_mask, boundary, gray
