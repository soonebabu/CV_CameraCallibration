"""Mask comparison utilities used to score our classical segmentation against a SAM2 mask."""

import cv2
import numpy as np


def to_binary_mask(img, thresh=127):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img > thresh).astype(np.uint8)


def keep_largest_component(binary_mask):
    """binary_mask: uint8 array of 0/1. Zeroes out every connected component except the largest."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return binary_mask
    # label 0 is background; pick the largest non-background component by area
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return (labels == largest_label).astype(np.uint8)


def iou(mask_a, mask_b):
    a, b = mask_a.astype(bool), mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def dice(mask_a, mask_b):
    a, b = mask_a.astype(bool), mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2 * inter / denom) if denom > 0 else 0.0


def overlay_comparison(shape_hw, mask_ours, mask_sam2):
    """White = agreement, green = ours only, red (magenta-ish) = SAM2 only. Returns a BGR image."""
    vis = np.zeros((*shape_hw, 3), dtype=np.uint8)
    a, b = mask_ours.astype(bool), mask_sam2.astype(bool)
    both = np.logical_and(a, b)
    only_ours = np.logical_and(a, np.logical_not(b))
    only_sam2 = np.logical_and(b, np.logical_not(a))
    vis[both] = (255, 255, 255)
    vis[only_ours] = (0, 200, 0)       # BGR green
    vis[only_sam2] = (0, 0, 220)       # BGR red
    return vis


def mask_to_display(mask):
    return (mask.astype(np.uint8) * 255)


def boundary_overlay(img_bgr, contour, color=(0, 255, 0), thickness=2):
    vis = img_bgr.copy()
    if contour is not None:
        cv2.drawContours(vis, [contour], -1, color, thickness)
    return vis
