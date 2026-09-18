"""
Classical (non-ML, non-DL) human boundary segmentation for regular color photos.

Uses OpenCV's GrabCut - an iterative graph-cut / Gaussian-mixture energy
minimization over a *single* image, seeded with a foreground rectangle. It is
not a trained/pretrained classifier: there is no model, no weights, and no
dataset involved, so it is allowed under the assignment's "no ML/DL" rule
(unlike, say, an HOG+SVM detector or a pretrained network).
"""

import cv2
import numpy as np

from .metrics import keep_largest_component


def default_rect(img_shape, margin_frac=0.05):
    """Whole-image rectangle inset by a margin, used when no manual box is given."""
    h, w = img_shape[:2]
    mx, my = int(w * margin_frac), int(h * margin_frac)
    return (mx, my, w - 2 * mx, h - 2 * my)


def grabcut_segment(img_bgr, rect, iterations=5):
    """
    rect: (x, y, w, h) foreground seed box.
    Returns (binary_mask uint8 {0,1}, largest_contour or None).
    """
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    if binary_mask.sum() > 0:
        binary_mask = keep_largest_component(binary_mask)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = max(contours, key=cv2.contourArea) if contours else None

    return binary_mask, boundary
