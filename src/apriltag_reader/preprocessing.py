"""Image preprocessing helpers."""

import cv2
import numpy as np


def apply_clahe(gray, clip_limit=3.0, grid=(8, 8)):
    # Improve local contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    return clahe.apply(gray)


def unsharp_mask(gray, amount=1.0, radius=3):
    # Sharpen the grayscale image using an unsharp mask
    blur = cv2.GaussianBlur(gray, (0, 0), radius)
    sharp = cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def preprocess(img_bgr, use_clahe, use_sharpen, upscale=1.0):
    # Convert image to grayscale if it is still in color
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()

    # The detector only accepts 8-bit images, so bring 16-bit frames down to 8-bit
    if gray.dtype == np.uint16:
        gray = (gray >> 8).astype(np.uint8)

    # Optionally upscale or downscale before detection to help with small tags
    if upscale and upscale != 1.0:
        h, w = gray.shape[:2]
        interp = cv2.INTER_CUBIC if upscale > 1.0 else cv2.INTER_AREA
        gray = cv2.resize(gray, (int(w * upscale), int(h * upscale)), interpolation=interp)

    # Optionally apply contrast enhancement
    if use_clahe:
        gray = apply_clahe(gray, clip_limit=3.0, grid=(8, 8))

    # Optionally sharpen the image
    if use_sharpen:
        gray = unsharp_mask(gray, amount=1.0, radius=1.5)

    gray = np.ascontiguousarray(gray)

    return gray


def canny_edges(gray):
    # Compute an edge image for quick debugging
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
    return edges
