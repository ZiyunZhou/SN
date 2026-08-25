# -*- coding: utf-8 -*-
"""
@author: zhouziyun
"""

import numpy as np
import os
from src import visualization, image_segmentation
import pandas as pd


def Nw(G):
    """Compute Neighborhood watch Nw = s / k for each node."""

    degrees = dict(G.degree())
    strengths = dict(G.degree(weight="weight"))

    nw = {
        n: strengths[n] / degrees[n] if degrees[n] > 0 else np.nan
        for n in G.nodes()
    }

    return nw


def analyse_network_metrics(G, region_info, img, weight_type="proximity", save_root="./results/network"):
    """Analyse and plot network metrics."""

    save_dir = os.path.join(save_root, weight_type)

    # Basic metrics
    degrees = dict(G.degree())
    strengths = dict(G.degree(weight="weight"))
    areas = {reg["id"]: reg["area"] for reg in region_info}
    perimeters = {reg["id"]: image_segmentation.contour_perimeter(reg["coords"]) for reg in region_info}

    # DataFrame
    nodes = list(G.nodes())
    df = pd.DataFrame({
        "id": nodes,
        "area": [areas[n] for n in nodes],
        "perimeter": [perimeters[n] for n in nodes],
        "degree": [degrees[n] for n in nodes],
        "strength": [strengths[n] for n in nodes]
    })

    df["Nw"] = df["strength"] / df["degree"]
    df["degree_area"] = df["degree"] / df["area"]
    df["strength_area"] = df["strength"] / df["area"]
    df["degree_perimeter"] = df["degree"] / df["perimeter"]
    df["strength_perimeter"] = df["strength"] / df["perimeter"]

    # wij
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    visualization.plot_net_edges_sepcolor(G, region_info, img, save_path=f"{save_dir}/wij.png")
    visualization.plot_metric_distribution(weights, xlabel="$w_{ij}$", bins=60, save_path=f"{save_dir}/wij_distribution.png")

    # Degree
    visualization.plot_metric_distribution(df["degree"], xlabel="$k$", discrete=True, save_path=f"{save_dir}/degree_distribution.png")
    visualization.plot_metric_poisson(df["degree"], xlabel="$k$", ylabel="$P(k)$", save_path=f"{save_dir}/degree.png")
    visualization.plot_metric_spatial(G, degrees, region_info, img, label="k", save_path=f"{save_dir}/degree_spatial.png")

    # Strength
    visualization.plot_metric_distribution(df["strength"], xlabel="$s$", bins=60, save_path=f"{save_dir}/strength_distribution.png")
    visualization.plot_metric_spatial(G, strengths, region_info, img, label="s", save_path=f"{save_dir}/strength_spatial.png")

    # Area-normalised
    visualization.plot_metric_distribution(df["degree_area"], xlabel="$k/A$", bins=60, log_y=True, save_path=f"{save_dir}/degree_area_distribution.png")
    visualization.plot_metric_distribution(df["strength_area"], xlabel="$s/A$", bins=60, save_path=f"{save_dir}/strength_area_distribution.png")
    visualization.plot_metric_spatial(G, dict(zip(df["id"], df["degree_area"])), region_info, img, label="k/A", 
                                      decimals=3, grouping="quantile", save_path=f"{save_dir}/degree_area_spatial.png")
    visualization.plot_metric_spatial(G, dict(zip(df["id"], df["strength_area"])), region_info, img, label="s/A",
                                      grouping="quantile", save_path=f"{save_dir}/strength_area_spatial.png")

    # Perimeter-normalised
    visualization.plot_metric_distribution(df["degree_perimeter"], xlabel="$k/P$", bins=60, save_path=f"{save_dir}/degree_perimeter_distribution.png")
    visualization.plot_metric_distribution(df["strength_perimeter"], xlabel="$s/P$", bins=60, save_path=f"{save_dir}/strength_perimeter_distribution.png")
    visualization.plot_metric_spatial(G, dict(zip(df["id"], df["degree_perimeter"])), region_info, img, label="k/P", 
                                      decimals=3, grouping="quantile", save_path=f"{save_dir}/degree_perimeter_spatial.png")
    visualization.plot_metric_spatial(G, dict(zip(df["id"], df["strength_perimeter"])), region_info, img, label="s/P",
                                      grouping="quantile", save_path=f"{save_dir}/strength_perimeter_spatial.png")

    return df


def metric_scalings(metrics_df, weight_type="proximity",
                    scalings=("kA", "sA", "kP", "sP", "sk", "sA_kA", "sP_kP"),
                    save_root="./results/network"):
    """Plot selected scaling relationships between node metrics."""

    save_dir = os.path.join(save_root, weight_type)

    scaling_pairs = {
        "kA": ("area", "degree", "$A$", "$k$"),
        "sA": ("area", "strength", "$A$", "$s$"),
        "kP": ("perimeter", "degree", "$P$", "$k$"),
        "sP": ("perimeter", "strength", "$P$", "$s$"),
        "sk": ("degree", "strength", "$k$", "$s$"),
        "sA_kA": ("degree_area", "strength_area", "$k/A$", "$s/A$"),
        "sP_kP": ("degree_perimeter", "strength_perimeter", "$k/P$", "$s/P$")
    }

    for name in scalings:
        x_col, y_col, xlabel, ylabel = scaling_pairs[name]
        x = metrics_df[x_col]
        y = metrics_df[y_col]

        visualization.plot_scaling(
            x, y, xlabel=xlabel, ylabel=ylabel,
            save_path=os.path.join(save_dir, f"scaling_{name}_logscale.png")
        )

        visualization.plot_scaling(
            x, y, xlabel=xlabel, ylabel=ylabel, log_scale=False,
            save_path=os.path.join(save_dir, f"scaling_{name}.png")
        )

