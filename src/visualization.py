# -*- coding: utf-8 -*-
"""
@author: zhouziyun
"""

import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter
from scipy.stats import poisson
import cv2
import networkx as nx
import matplotlib as mpl
from matplotlib.patches import Patch
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_length_distribution(data1, data2, label1, label2, xlabel, save_path=None, log=False, bins=50):
    plt.figure(figsize=(6, 5))
    plt.hist(data1, bins=bins, color="lightgray", alpha=0.8, label=label1)
    plt.hist(data2, bins=bins, color="dimgray", alpha=0.8, label=label2)

    if log:
        plt.yscale("log")

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Counts", fontsize=14)
    plt.legend(frameon=False, fontsize=13)
    plt.tight_layout()

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.close()


def plot_net_edges_sepcolor(G, region_info, img, save_path=None, top_percent=100):
    """Visualize high-weight edges using six discrete colors."""

    colors = ["blue", "cyan", "greenyellow", "yellow", "orange", "red"]
    pos = nx.get_node_attributes(G, "pos")

    edges = list(G.edges(data=True))
    if not edges:
        return

    weights = np.array([d["weight"] for _, _, d in edges], dtype=float)
    threshold = np.percentile(weights, 100 - top_percent)
    high_edges = [(u, v, d) for u, v, d in edges if d["weight"] >= threshold]

    if not high_edges:
        return

    high_weights = np.array([d["weight"] for _, _, d in high_edges], dtype=float)
    bounds = np.linspace(high_weights.min(), high_weights.max(), 7)

    edge_colors = []
    for _, _, d in high_edges:
        idx = min(np.searchsorted(bounds[1:], d["weight"], side="right"), 5)
        edge_colors.append(colors[idx])

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.3], wspace=0.02)

    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_aspect("equal")
    ax.axis("off")

    for reg in region_info:
        cnt = reg["coords"]
        poly = mpl.patches.Polygon(cnt, closed=True, facecolor="lightgray", edgecolor="gray", linewidth=0.8, zorder=1)
        ax.add_patch(poly)

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v) for u, v, _ in high_edges],
        ax=ax,
        edge_color=edge_colors,
        width=1.5
    )

    legend_labels = [
        rf"$w_{{ij}} < {bounds[1]:.0f}$",
        rf"${bounds[1]:.0f} \leq w_{{ij}} < {bounds[2]:.0f}$",
        rf"${bounds[2]:.0f} \leq w_{{ij}} < {bounds[3]:.0f}$",
        rf"${bounds[3]:.0f} \leq w_{{ij}} < {bounds[4]:.0f}$",
        rf"${bounds[4]:.0f} \leq w_{{ij}} < {bounds[5]:.0f}$",
        rf"$w_{{ij}} \geq {bounds[5]:.0f}$"
    ]

    legend_elements = [Patch(facecolor=colors[i], edgecolor="black", label=legend_labels[i]) for i in range(6)]

    ax_leg.legend(
        handles=legend_elements,
        # bbox_to_anchor=(0.1, 0.5),
        loc="center left",
        frameon=False,
        fontsize=12,
        borderpad=0,
        labelspacing=0.8,
        handlelength=1.2,
        handleheight=0.8
    )

    if save_path:
        if top_percent != 100:
            root, ext = os.path.splitext(save_path)
            save_path = f"{root}_top{top_percent}pct{ext}"
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, pad_inches=0, dpi=300)

    plt.close(fig)
    
    
def plot_metric_spatial1(G, metric_dict, region_info, img, label="metric", 
                        save_path=None, decimals=0, grouping="equal"):
    """Visualize spatial distribution of a node metric using six discrete colors."""

    colors = ["blue", "cyan", "greenyellow", "yellow", "orange", "red"]
    pos = nx.get_node_attributes(G, "pos")
    region_dict = {reg["id"]: reg for reg in region_info}

    valid_nodes = [n for n in G.nodes() if n in metric_dict and not np.isnan(metric_dict[n])]
    if not valid_nodes:
        return

    values = np.array([metric_dict[n] for n in valid_nodes], dtype=float)
    if grouping == "quantile":
        bounds = np.percentile(values, np.linspace(0, 100, 7))
    else:
        bounds = np.linspace(values.min(), values.max(), 7)

    node_colors = {}
    for n in valid_nodes:
        idx = min(np.searchsorted(bounds[1:], metric_dict[n], side="right"), 5)
        node_colors[n] = colors[idx]

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.3], wspace=0.02)

    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Network edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", width=0.8, alpha=0.35)

    # Stromatolite polygons
    for n in valid_nodes:
        cnt = region_dict[n]["coords"]
        poly = mpl.patches.Polygon(
            cnt,
            closed=True,
            facecolor=node_colors[n],
            edgecolor="gray",
            linewidth=0.8,
            zorder=2
        )
        ax.add_patch(poly)

    # Legend
    fmt = f".{decimals}f"

    legend_labels = [
        rf"${label} < {bounds[1]:{fmt}}$",
        rf"${bounds[1]:{fmt}} \leq {label} < {bounds[2]:{fmt}}$",
        rf"${bounds[2]:{fmt}} \leq {label} < {bounds[3]:{fmt}}$",
        rf"${bounds[3]:{fmt}} \leq {label} < {bounds[4]:{fmt}}$",
        rf"${bounds[4]:{fmt}} \leq {label} < {bounds[5]:{fmt}}$",
        rf"${label} \geq {bounds[5]:{fmt}}$"
    ]

    legend_elements = [
        Patch(facecolor=colors[i], edgecolor="black", linewidth=0.8, label=legend_labels[i])
        for i in range(6)
    ]

    ax_leg.legend(
        handles=legend_elements,
        loc="center left",
        frameon=False,
        fontsize=12,
        borderpad=0,
        labelspacing=0.8,
        handlelength=1.2,
        handleheight=0.8,
        handletextpad=0.8
    )

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, dpi=300, pad_inches=0)

    plt.close(fig)


def plot_metric_spatial(G, metric_dict, region_info, img, label="metric", save_path=None, decimals=0, grouping="equal", color_style="rainbow"):
    """Visualize spatial distribution of a node metric."""

    pos = nx.get_node_attributes(G, "pos")
    region_dict = {reg["id"]: reg for reg in region_info}
    valid_nodes = [n for n in G.nodes() if n in metric_dict and not np.isnan(metric_dict[n])]

    if not valid_nodes:
        return

    values = np.array([metric_dict[n] for n in valid_nodes], dtype=float)

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.3], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_aspect("equal")
    ax.axis("off")

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", width=0.8, alpha=0.35)

    # Six discrete rainbow colors
    if color_style == "rainbow":
        colors = ["blue", "cyan", "greenyellow", "yellow", "orange", "red"]
        bounds = np.percentile(values, np.linspace(0, 100, 7)) if grouping == "quantile" else np.linspace(values.min(), values.max(), 7)

        for n in valid_nodes:
            idx = min(np.searchsorted(bounds[1:], metric_dict[n], side="right"), 5)
            poly = mpl.patches.Polygon(region_dict[n]["coords"], closed=True, facecolor=colors[idx], edgecolor="gray", linewidth=0.8, zorder=2)
            ax.add_patch(poly)

        fmt = f".{decimals}f"
        legend_labels = [
            rf"${label} < {bounds[1]:{fmt}}$",
            rf"${bounds[1]:{fmt}} \leq {label} < {bounds[2]:{fmt}}$",
            rf"${bounds[2]:{fmt}} \leq {label} < {bounds[3]:{fmt}}$",
            rf"${bounds[3]:{fmt}} \leq {label} < {bounds[4]:{fmt}}$",
            rf"${bounds[4]:{fmt}} \leq {label} < {bounds[5]:{fmt}}$",
            rf"${label} \geq {bounds[5]:{fmt}}$"
        ]

        legend_elements = [Patch(facecolor=colors[i], edgecolor="black", linewidth=0.8, label=legend_labels[i]) for i in range(6)]
        ax_leg.legend(handles=legend_elements, loc="center left", frameon=False, fontsize=12, borderpad=0, labelspacing=0.8, handlelength=1.2, handleheight=0.8, handletextpad=0.8)

    # Continuous blue → green → red
    elif color_style == "rgb":
        blue_rgb = (0, 0, 1)
        green_rgb = (0, 1, 0)
        red_rgb = (1, 0, 0)
        
        cmap = mcolors.LinearSegmentedColormap.from_list("rgb_scale", [blue_rgb, green_rgb, red_rgb])
        # cmap = mcolors.LinearSegmentedColormap.from_list("rgb_scale", ["blue", "green", "red"])
        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

        for n in valid_nodes:
            poly = mpl.patches.Polygon(region_dict[n]["coords"], closed=True, facecolor=cmap(norm(metric_dict[n])), edgecolor="gray", linewidth=0.8, zorder=2)
            ax.add_patch(poly)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_leg, fraction=0.25, shrink=0.65)
        cbar.ax.set_title(rf"${label}$", fontsize=12, pad=8)
        cbar.ax.tick_params(labelsize=11)

    else:
        raise ValueError("color_style must be 'rainbow' or 'rgb'")

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=300, pad_inches=0)

    plt.close(fig)

def plot_metric_poisson(values, xlabel="Value", ylabel="P(Value)", title=None, save_path=None):
    """Plot empirical distribution with a Poisson fit."""

    values = np.asarray(values)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return

    N = len(values)

    counts = Counter(values)
    x_vals = np.array(sorted(counts.keys()))
    P = np.array([counts[x] for x in x_vals]) / N

    lambda_poisson = values.mean()
    poisson_std = np.sqrt(lambda_poisson)
    x_fit = np.arange(x_vals.min(), x_vals.max() + 1)
    P_fit = poisson.pmf(x_fit, lambda_poisson)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.set_position([0.16, 0.13, 0.78, 0.78])

    ax.bar(x_vals, P, width=0.8, color="gray", edgecolor="white", linewidth=0.8, alpha=0.9)
    ax.plot(x_fit, P_fit, "k--o", linewidth=1.8, markersize=4, label=f"Poisson fit\n(mean = {lambda_poisson:.2f}, std = {poisson_std:.2f})")

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(np.arange(int(x_vals.min()), int(x_vals.max()) + 1))
    ax.tick_params(axis="both", labelsize=12)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    # ax.text(0.65, 0.65, f"N = {N}", transform=ax.transAxes, fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)

    ax.legend(fontsize=11, frameon=False)

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, dpi=300, pad_inches=0.1)

    plt.close(fig)


def plot_metric_distribution(values, xlabel="Value", ylabel="Counts", title=None,
                             save_path=None, bins=30, color="gray",
                             discrete=False, log_x=False, log_y=False):
    """Plot and save the distribution of a metric."""

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return

    if discrete:
        bins = np.arange(values.min() - 0.5, values.max() + 1.5, 1)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.set_position([0.16, 0.13, 0.78, 0.78])

    ax.hist(values, bins=bins, color=color, edgecolor="white", alpha=0.9)

    if discrete:
        ax.set_xticks(np.arange(int(values.min()), int(values.max()) + 1))

    if log_x:
        ax.set_xscale("log")

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    if title:
        ax.set_title(title, fontsize=14)

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, pad_inches=0.1)

    plt.close(fig)


def fit_scaling(x, y):
    """Fit a power-law relationship in log-log space."""

    if isinstance(x, dict) and isinstance(y, dict):
        nodes = [n for n in x if n in y]
        x = np.array([x[n] for n in nodes], dtype=float)
        y = np.array([y[n] for n in nodes], dtype=float)
    else:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

    mask = (x > 0) & (y > 0) & ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    logx = np.log10(x)
    logy = np.log10(y)

    slope, intercept = np.polyfit(logx, logy, 1)
    fit = slope * logx + intercept

    ss_res = np.sum((logy - fit) ** 2)
    ss_tot = np.sum((logy - logy.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return x, y, slope, intercept, r2


def plot_scaling(x, y, xlabel="x", ylabel="y", title=None, save_path=None, log_scale=True):
    """Plot a power-law scaling relationship with optional log-log axes."""

    x, y, slope, intercept, r2 = fit_scaling(x, y)

    x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    y_fit = 10**intercept * x_fit**slope

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_position([0.18, 0.16, 0.76, 0.76])

    ax.scatter(x, y, s=40, color="dodgerblue", edgecolor="black", alpha=0.8)
    ax.plot(x_fit, y_fit, color="red", linestyle="--", linewidth=2)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    ax.text(
        0.05, 0.93,
        rf"$\alpha$ = {slope:.2f}" + "\n" + rf"$R^2$ = {r2:.2f}",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top"
    )

    if title:
        ax.set_title(title, fontsize=14)

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=300, pad_inches=0.1)

    plt.close(fig)

    return slope, r2


def group_by_area(areas, n_groups=5):
    """Group stromatolites by area quantiles."""

    if isinstance(areas, dict):
        nodes = list(areas.keys())
        values = np.array([areas[n] for n in nodes], dtype=float)
    else:
        nodes = None
        values = np.asarray(areas, dtype=float)

    bounds = np.quantile(values, np.linspace(0, 1, n_groups + 1))
    groups = np.digitize(values, bounds[1:-1], right=True)

    if nodes is not None:
        groups = {n: groups[i] for i, n in enumerate(nodes)}

    return groups, bounds


def plot_area_groups_spatial(area_groups, region_info, img, n_groups=5, save_path=None):
    """Plot the spatial distribution of area groups."""

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(n_groups)]
    region_dict = {reg["id"]: reg for reg in region_info}

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.25], wspace=0.02)

    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_aspect("equal")
    ax.axis("off")

    for n, group in area_groups.items():
        cnt = region_dict[n]["coords"]
        poly = mpl.patches.Polygon(
            cnt,
            closed=True,
            facecolor=colors[group % len(colors)],
            edgecolor="gray",
            linewidth=0.8
        )
        ax.add_patch(poly)

    legend_elements = [
        Patch(facecolor=colors[i % len(colors)], edgecolor="black", label=f"Group {i + 1}")
        for i in range(n_groups)
    ]

    ax_leg.legend(
        handles=legend_elements,
        loc="center left",
        frameon=False,
        fontsize=12,
        borderpad=0,
        labelspacing=0.8,
        handlelength=1.2,
        handleheight=0.8
    )

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=300, pad_inches=0)

    plt.close(fig)


def plot_group_scaling(x, y, groups, n_groups=5, xlabel="x", ylabel="y", log_scale=True, save_path=None):
    """Plot scaling relationships for different groups."""

    nodes = [n for n in x if n in y and n in groups]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(n_groups)]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_position([0.18, 0.16, 0.76, 0.76])

    for group in sorted(set(groups.values())):
        group_nodes = [n for n in nodes if groups[n] == group]

        xg = np.array([x[n] for n in group_nodes], dtype=float)
        yg = np.array([y[n] for n in group_nodes], dtype=float)

        mask = (xg > 0) & (yg > 0)
        xg = xg[mask]
        yg = yg[mask]

        if len(xg) < 2:
            continue

        _, _, slope, intercept, r2 = fit_scaling(xg, yg)

        x_fit = np.logspace(np.log10(xg.min()), np.log10(xg.max()), 100)
        y_fit = 10**intercept * x_fit**slope

        color = colors[group % len(colors)]

        ax.scatter(xg, yg, s=35, color=color, edgecolor="black", alpha=0.8,
            label=rf"Group {group + 1} ($\alpha$={slope:.2f}, $R^2$={r2:.2f})")
        ax.plot(x_fit, y_fit, color=color, linewidth=2, linestyle="--",)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(frameon=False, fontsize=9, labelspacing=0.5, handletextpad=0.5)

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=300, pad_inches=0.1)

    plt.close(fig)




