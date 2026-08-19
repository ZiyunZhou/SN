# -*- coding: utf-8 -*-
"""
@author: zhouziyun
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import os
import geopandas as gpd
from shapely.geometry import Polygon


def watershed(img, sure_fg, unknown):
    """Perform watershed segmentation using foreground and unknown markers."""
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img_ws = img.copy()
    img_ws[markers == -1] = [255, 0, 0]
    
    return markers
    

def extract_contours_cleancolor(img, markers, img_alpha, stroma_alpha, 
                                save_path="./results/image_segmentation/final_contours.png"):
    """Extract contours, centroids and areas, and visualize them on a transparent background image."""

    region_info = []

    # markers: -1 = boundary, 0 = unknown, 1 = background, >=2 = segmented objects
    for label in np.unique(markers):
        if label <= 1:
            continue

        mask = (markers == label).astype(np.uint8)

        # Full-resolution contour
        cnts_full, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts_full:
            continue

        cnt_full = max(cnts_full, key=cv2.contourArea)
        area = cv2.contourArea(cnt_full)
        if area < 1:
            continue

        # Centroid
        M = cv2.moments(cnt_full)
        cx = int(M["m10"] / M["m00"]) if M["m00"] else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] else 0
        
        sid = len(region_info) + 1
        
        # Save info
        region_info.append({
            "id": sid,
            "coords": cnt_full[:, 0, :],
            "centroid": (cx, cy),
            "area": area})

    # Visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=img_alpha)

    for info in region_info:
        pts = info["coords"]
        cx, cy = info["centroid"]
        plt.fill(pts[:, 0], pts[:, 1], color="lightgrey", alpha=stroma_alpha, zorder=0)
        plt.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), 
                 color="blue", lw=0.5, alpha=1, zorder=1)
        plt.scatter(cx, cy, s=3, color="lime", alpha=1, zorder=2)

    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    
    legend_elements = [
        Patch(facecolor="lightgrey", edgecolor="none", label="Stromatolite"),
        Line2D([0], [0], color="blue", lw=0.8, label="Contour"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="lime", markeredgecolor="lime", markersize=6, label="Centroid")
    ]
    
    fig = plt.figure(figsize=(3, 2.5))
    fig.legend(handles=legend_elements, loc="center", ncol=1, frameon=False, fontsize=15)
    plt.axis("off")
    plt.savefig(save_path.replace(".png", "_legend.png"), bbox_inches="tight", pad_inches=0)
    plt.close()
    return region_info


def gama_shapefile(img, region_info, save_path="./results/image_segmentation/stroma_space.shp"):
    """Export segmented stromatolite regions as a shapefile for GAMA."""

    polygons = []
    img_height = img.shape[0]

    for region in region_info:
        coords = region["coords"]
        flipped_coords = [(x, img_height - y) for x, y in coords]

        if len(flipped_coords) >= 3:
            polygons.append(Polygon(flipped_coords))

    folder = os.path.dirname(save_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    gdf = gpd.GeoDataFrame(geometry=polygons)
    gdf.to_file(save_path)

    return gdf
