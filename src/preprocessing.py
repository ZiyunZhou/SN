# -*- coding: utf-8 -*-
"""
@author: zhouziyun
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def read_image(img_path):
    
    img = cv2.imread(img_path)
    
    return img


def get_meter_per_pixel(image, real_length_m, save_path="./results/image_segmentation/scale_bar_detection.png"):
    """Detect the horizontal scale bar and return meters per pixel."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img = image.shape[0]
    candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if y >= 0.6 * h_img and w > 50 and h < 0.3 * w:
            candidates.append((w, x, y, h))

    if not candidates:
        raise ValueError("Scale bar not detected.")

    w, x, y, h = max(candidates)

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        vis = image.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(save_path, vis)

    return real_length_m / w


def grayscale(img, save_path="./results/image_segmentation/grayscale.png"):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        cv2.imwrite(save_path, gray)
        
    return gray


def Gaussian_blur(gray, save_path="./results/image_segmentation/Gaussian_blur.png"):
    
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, gray_blur)

    return gray_blur


def Otsu(gray_blur, black_thresh, 
         save_path="./results/image_segmentation/Otsu.png"):
    """Remove black background based on intensity and apply Otsu binarization."""
    
    mask = gray_blur > black_thresh
    filtered = gray_blur.copy() # grayscale image with dark background masked out.
    filtered[~mask] = 255
    # binary: binary image obtained by Otsu thresholding.
    # otsu_t: global threshold value computed by Otsu’s method.
    otsu_t, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, binary)
    
    return binary

def morphological_cleanup(binary, iteration_open, iteration_close, 
                          save_path="./results/image_segmentation/morphological_cleanup.png"):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iteration_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iteration_close)
    sure_bg = cv2.dilate(closed, kernel, iterations=1)

    # Regions modified by opening and closing
    removed = (binary > 0) & (opened == 0)
    added = (opened == 0) & (closed > 0)

    # Convert binary image to RGB for colored display
    vis = np.full((*binary.shape, 3), 255, dtype=np.uint8) # white background
    vis[binary > 0] = [160, 160, 160]  # gray structures
    vis[removed] = [255, 0, 0] # red: removed noise
    vis[added] = [0, 255, 255] # blue: filled gaps
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
    plt.imsave(save_path, vis)

    return closed, sure_bg


def extract_foreground_unknown1(closed, sure_bg, threshold_ratio, dist_save_path, unknown_save_path):
    """Compute distance transform, extract foreground, and define unknown regions."""
    dist_transform = cv2.distanceTransform(closed, cv2.DIST_L2, 3)
    _, sure_fg = cv2.threshold(dist_transform,
        threshold_ratio * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    fig_dist = plt.figure(figsize=(6, 6))
    plt.imshow(dist_transform, cmap='inferno')
    plt.title("Distance Transform", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(dist_save_path)
    plt.close(fig_dist)
    
    fig_unknown = plt.figure(figsize=(6, 6))
    plt.imshow(unknown, cmap='gray')
    plt.title("Unknown Region", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(unknown_save_path)
    plt.close(fig_unknown)
    
    return sure_fg, unknown


def extract_foreground_unknown(closed, sure_bg, threshold_ratio, 
                               dist_save_path="./results/image_segmentation/distance_trans.png", 
                               unknown_save_path=None):
    """Extract sure foreground and unknown regions using distance transform."""

    dist_transform = cv2.distanceTransform(closed, cv2.DIST_L2, 3)
    _, sure_fg = cv2.threshold(dist_transform, threshold_ratio * dist_transform.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    for img, path, cmap in [(dist_transform, dist_save_path, "inferno"), 
                            (unknown, unknown_save_path, "gray")]:
        if path:
            folder = os.path.dirname(path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            plt.imsave(path, img, cmap=cmap)

    return sure_fg, unknown


def plot_watershed_regions(sure_bg, sure_fg, unknown, 
                           save_path="./results/image_segmentation/watershed_regions.png"):
    bg_mask = sure_bg == 0
    unknown_mask = unknown > 0
    fg_mask = sure_fg > 0

    region_map = np.zeros(sure_bg.shape, dtype=np.uint8)
    region_map[bg_mask] = 0
    region_map[unknown_mask] = 1
    region_map[fg_mask] = 2

    cmap = ListedColormap(["white", "#A0A0A0", "lightgray"])
    plt.figure(figsize=(8, 6))
    plt.imshow(region_map, cmap=cmap, interpolation="nearest")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    legend_elements = [
        Patch(facecolor="white", edgecolor="black", label="Background"),
        Patch(facecolor="#A0A0A0", edgecolor="black", label="Unknown Region"),
        Patch(facecolor="lightgray", edgecolor="black", label="Sure Foreground")
    ]

    fig = plt.figure(figsize=(3, 2.5))
    fig.legend(handles=legend_elements, loc="center", frameon=False, fontsize=15)
    plt.axis("off")
    plt.savefig(save_path.replace(".png", "_legend.png"), bbox_inches="tight", pad_inches=0)
    plt.close()
   
