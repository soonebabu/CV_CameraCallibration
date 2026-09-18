"""
Image blurring via spatial filtering and its Fourier-domain equivalent.

The core claim under test (the convolution theorem):

    f(x,y) * h(x,y)  <-->  F(u,v) . H(u,v)

i.e. convolving an image with a kernel in the spatial domain gives the exact
same result as multiplying their Fourier transforms and taking the inverse
transform. ``blur_compare`` computes the blur both ways on the same image and
returns both results plus the numerical difference between them, so the
equivalence can be checked directly rather than just asserted.
"""

import cv2
import numpy as np


def gaussian_kernel_2d(ksize, sigma):
    if ksize % 2 == 0:
        ksize += 1
    ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def box_kernel_2d(ksize):
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.ones((ksize, ksize), dtype=np.float64)
    kernel /= kernel.sum()
    return kernel


def spatial_convolve(channel, kernel):
    """True convolution (not correlation) via cv2.filter2D, zero-padded border."""
    flipped = kernel[::-1, ::-1]
    return cv2.filter2D(channel.astype(np.float64), -1, flipped, borderType=cv2.BORDER_CONSTANT)


def frequency_convolve(channel, kernel):
    """
    Linear convolution computed as a padded circular convolution in the
    Fourier domain, matching spatial_convolve's zero-padded border exactly.

    A plain FFT multiply on the original image size gives *circular*
    convolution (edges wrap around). To get the same result as the
    zero-padded spatial convolution, the image is first zero-padded by the
    kernel's radius on every side; that padding gives the wraparound enough
    "room" to land entirely inside the pad, so the center region of the
    result equals true linear convolution. The kernel is placed at the
    array origin (np.roll) since FFT-based convolution assumes the kernel
    is centered at index (0, 0), not at its visual center.
    """
    h, w = channel.shape
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2

    padded = cv2.copyMakeBorder(channel.astype(np.float64), pad_y, pad_y, pad_x, pad_x,
                                 cv2.BORDER_CONSTANT, value=0)
    ph, pw = padded.shape

    kernel_full = np.zeros((ph, pw), dtype=np.float64)
    kernel_full[:kh, :kw] = kernel
    kernel_full = np.roll(kernel_full, (-pad_y, -pad_x), axis=(0, 1))

    F_img = np.fft.fft2(padded)
    F_kernel = np.fft.fft2(kernel_full)
    full_result = np.fft.ifft2(F_img * F_kernel).real

    return full_result[pad_y:pad_y + h, pad_x:pad_x + w]


def magnitude_spectrum_image(channel):
    """Log-scaled, fftshifted magnitude spectrum, normalized to 0-255 for display."""
    F = np.fft.fftshift(np.fft.fft2(channel.astype(np.float64)))
    mag = np.log1p(np.abs(F))
    if mag.max() > 0:
        mag = (mag / mag.max()) * 255.0
    return mag.astype(np.uint8)


def kernel_frequency_response(kernel, shape):
    """Magnitude of the kernel's own frequency response, padded to `shape` (fftshifted)."""
    h, w = shape
    kh, kw = kernel.shape
    padded = np.zeros((h, w), dtype=np.float64)
    padded[:kh, :kw] = kernel
    padded = np.roll(padded, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    F = np.fft.fftshift(np.fft.fft2(padded))
    mag = np.abs(F)
    if mag.max() > 0:
        mag = (mag / mag.max()) * 255.0
    return mag.astype(np.uint8)


def normalize_for_display(arr):
    a = arr.astype(np.float64)
    if a.max() > 1e-12:
        a = (a / a.max()) * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def blur_compare(img_bgr, kernel_type="gaussian", ksize=15, sigma=3.0):
    """
    Runs spatial-domain and frequency-domain blurring on the same image with
    the same kernel and returns everything needed to display and validate
    the comparison.
    """
    if kernel_type == "box":
        kernel = box_kernel_2d(ksize)
    else:
        kernel = gaussian_kernel_2d(ksize, sigma)

    channels = cv2.split(img_bgr.astype(np.float64))
    spatial_channels, freq_channels = [], []
    for ch in channels:
        spatial_channels.append(spatial_convolve(ch, kernel))
        freq_channels.append(frequency_convolve(ch, kernel))

    spatial_result = cv2.merge(spatial_channels)
    freq_result = cv2.merge(freq_channels)
    diff = spatial_result - freq_result

    metrics = {
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "mse": float(np.mean(diff ** 2)),
        "psnr_db": _psnr(spatial_result, freq_result),
    }

    spatial_u8 = np.clip(spatial_result, 0, 255).astype(np.uint8)
    freq_u8 = np.clip(freq_result, 0, 255).astype(np.uint8)
    diff_vis = normalize_for_display(np.abs(diff))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    spectrum_original = magnitude_spectrum_image(gray)
    spectrum_blurred = magnitude_spectrum_image(cv2.cvtColor(spatial_u8, cv2.COLOR_BGR2GRAY))
    spectrum_kernel = kernel_frequency_response(kernel, gray.shape)

    return {
        "kernel": kernel,
        "spatial_u8": spatial_u8,
        "freq_u8": freq_u8,
        "diff_vis": diff_vis,
        "metrics": metrics,
        "spectrum_original": spectrum_original,
        "spectrum_blurred": spectrum_blurred,
        "spectrum_kernel": spectrum_kernel,
    }


def _psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return None
    return 10.0 * np.log10((255.0 ** 2) / mse)
