# -*- coding: utf-8 -*-
"""
@author: zhouziyun
"""

import numpy as np
from scipy.spatial import Delaunay
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import pandas as pd
import os
from src.visualization import plot_distribution


def collect_delaunay_points(region_info):
    """Collect full-resolution contour points from all structures."""

    all_points = []
    structure_ids = []
    point_meta = []

    for reg in region_info:
        sid = reg["id"]
        pts = reg["coords"]
        n_contour = len(pts)

        for idx, p in enumerate(pts):
            all_points.append(p)
            structure_ids.append(sid)
            point_meta.append({"sid": sid, "idx": idx, "n_contour": n_contour})

    return np.asarray(all_points), np.asarray(structure_ids), point_meta


def compute_delaunay(all_points, img, bg_alpha, save_path="./results/Delaunay/delaunay.png"):
    """Perform global Delaunay triangulation on all contour points."""
    
    tri = Delaunay(all_points)
    
    # plot
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=bg_alpha)
    # plt.axis("equal")
    plt.axis("off")
    plt.triplot(all_points[:, 0], all_points[:, 1], 
                tri.simplices, color='gray', lw=0.4)
    plt.scatter(all_points[:, 0], all_points[:, 1], 
                s=0.5, c='blue', alpha=1, edgecolors='none')
    
    plt.tight_layout()
    
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    
    plt.close()
    
    return tri


def filter_exterior_triangles(tri, all_points, region_info, 
                              img, bg_alpha, save_path="./results/Delaunay/exterior_triangles.png"):
    """Keep only triangles whose centroid lies outside all stromatolite polygons."""
    
    # Build polygons for each stromatolite
    polygons = []
    for reg in region_info:
        cnt = reg["coords"]
        if len(cnt) >= 3:
            polygons.append(Polygon(cnt))

    exterior_tri = []

    for simplex in tri.simplices:
        tri_coords = all_points[simplex]
        tri_poly = Polygon(tri_coords)
        centroid = tri_poly.centroid

        if not any(p.contains(centroid) for p in polygons):
            exterior_tri.append(simplex)
            
    # Visualize
    plt.figure(figsize=(6, 6))
    
    # original image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=bg_alpha, zorder=0)
    
    # exterior triangles
    for simplex in exterior_tri:
        pts = all_points[simplex]
        plt.plot(np.append(pts[:, 0], pts[0, 0]),
                 np.append(pts[:, 1], pts[0, 1]),
                 color='gray', lw=0.4, alpha=1, zorder=1)
    
    # contours
    plt.scatter(all_points[:, 0], all_points[:, 1], 
                s=0.5, c='blue', alpha=1, edgecolors='none', zorder=2)

    plt.axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0) # , dpi=300
    plt.close()

    return exterior_tri


def plot_edge_length_distribution(all_points, tri, exterior_triangles, point_meta, 
                                  save_dir="./results/Delaunay", 
                                  filename_prefix="edge_length", show_stats=False):
    """Compute and plot Delaunay edge-length distributions."""

    def get_triangle_edges(simplices):
        edges = set()
        for simplex in simplices:
            for i in range(3):
                edges.add(tuple(sorted((simplex[i], simplex[(i + 1) % 3]))))
        return np.array(list(edges))

    def is_contour_neighbor_edge(u, v):
        meta_u, meta_v = point_meta[u], point_meta[v]

        if meta_u["sid"] != meta_v["sid"]:
            return False

        diff = abs(meta_u["idx"] - meta_v["idx"])
        return diff == 1 or diff == meta_u["n_contour"] - 1

    def get_lengths(edges):
        return np.linalg.norm(all_points[edges[:, 0]] - all_points[edges[:, 1]], axis=1)

    def filter_lengths(edges, lengths):
        keep = [not is_contour_neighbor_edge(u, v) for u, v in edges]
        return lengths[keep]

    edges_all = get_triangle_edges(tri.simplices)
    edges_ext = get_triangle_edges(exterior_triangles)

    edge_lengths_all = get_lengths(edges_all)
    edge_lengths_ext = get_lengths(edges_ext)

    filtered_lengths_all = filter_lengths(edges_all, edge_lengths_all)
    filtered_lengths_ext = filter_lengths(edges_ext, edge_lengths_ext)

    if save_dir:
        plot_distribution(edge_lengths_all, edge_lengths_ext, "All edges", "Exterior edges", "Edge length", f"{save_dir}/{filename_prefix}_original_linear.png")
        plot_distribution(edge_lengths_all, edge_lengths_ext, "All edges", "Exterior edges", "Edge length", f"{save_dir}/{filename_prefix}_original_log.png", log=True)
        plot_distribution(filtered_lengths_all, filtered_lengths_ext, "All edges", "Exterior edges", "Edge length", f"{save_dir}/{filename_prefix}_filtered_linear.png")
        plot_distribution(filtered_lengths_all, filtered_lengths_ext, "All edges", "Exterior edges", "Edge length", f"{save_dir}/{filename_prefix}_filtered_log.png", log=True)

    if show_stats:
        print("\nDelaunay Edge-Length Statistics")
        print(f"All edges: {len(edge_lengths_all):,}")
        print(f"Exterior edges: {len(edge_lengths_ext):,}")
        print(f"Filtered all edges: {len(filtered_lengths_all):,}")
        print(f"Filtered exterior edges: {len(filtered_lengths_ext):,}")
    
        print("\nMean lengths:")
        print(f"Original all: {np.mean(edge_lengths_all):.2f}")
        print(f"Original exterior: {np.mean(edge_lengths_ext):.2f}")
        print(f"Filtered all: {np.mean(filtered_lengths_all):.2f}")
        print(f"Filtered exterior: {np.mean(filtered_lengths_ext):.2f}")

    return edge_lengths_all, edge_lengths_ext, filtered_lengths_all, filtered_lengths_ext


def compute_void_metrics(region_info, all_points, exterior_triangles, structure_ids, 
                         img, save_dir="./results/Delaunay", filename_prefix="void", rtol=1e-6, atol=1e-6):
    """Compute total and split void metrics for each stromatolite."""

    # Classify each exterior triangle once
    triangle_info = []

    for simplex in exterior_triangles:
        ids = np.unique(structure_ids[simplex])
        area = Polygon(all_points[simplex]).area

        if len(ids) == 1:
            category = "one_stromatolite"
        elif len(ids) == 2:
            category = "two_stromatolite"
        elif len(ids) == 3:
            category = "three_stromatolite"
        else:
            continue

        triangle_info.append({"simplex": simplex, "ids": ids, "area": area, "category": category})

    global_pairwise = [t["simplex"] for t in triangle_info if t["category"] == "two_stromatolite"]
    global_multi = [t["simplex"] for t in triangle_info if t["category"] == "three_stromatolite"]
    global_isolated = [t["simplex"] for t in triangle_info if t["category"] == "one_stromatolite"]

    rows = []

    # Compute void metrics for each stromatolite
    for reg in region_info:
        sid = reg["id"]
        cx, cy = reg["centroid"]

        target_tris = [t for t in triangle_info if sid in t["ids"]]

        pairwise = [t for t in target_tris if t["category"] == "two_stromatolite"]
        multi = [t for t in target_tris if t["category"] == "three_stromatolite"]
        isolated = [t for t in target_tris if t["category"] == "one_stromatolite"]

        pairwise_area = sum(t["area"] for t in pairwise)
        multi_area = sum(t["area"] for t in multi)
        isolated_area = sum(t["area"] for t in isolated)
        void_area = pairwise_area + multi_area + isolated_area

        total_area = sum(t["area"] for t in target_tris)
        if not np.isclose(void_area, total_area, rtol=rtol, atol=atol):
            print(f"[sid {sid}] void area mismatch")

        rows.append({
            "structure_id": sid,
            "centroid_x": cx,
            "centroid_y": cy,
            "area": reg["area"],
            "perimeter_points": len(reg["coords"]),
            "void_area": void_area,
            "pairwise_area": pairwise_area,
            "multi_area": multi_area,
            "isolated_area": isolated_area,
            "pairwise_triangles": len(pairwise),
            "multi_triangles": len(multi),
            "isolated_triangles": len(isolated)
        })

    df = pd.DataFrame(rows)

    # Normalized metrics
    df["Np"] = df["void_area"] / df["perimeter_points"]
    df["Na"] = df["void_area"] / df["area"]

    n_triangles = df["pairwise_triangles"] + df["multi_triangles"] + df["isolated_triangles"]
    df["Nt"] = np.where(n_triangles > 0, df["void_area"] / n_triangles, np.nan)

    # Visualization
    if img is not None and save_dir:
        os.makedirs(save_dir, exist_ok=True)

        def draw_triangle_map(simplices, facecolor, edgecolor, fill_alpha, outname):
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0, zorder=0)

            for simplex in exterior_triangles:
                pts = all_points[simplex]
                plt.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color="gray", lw=0.2, alpha=0.25, zorder=1)

            for simplex in simplices:
                pts = all_points[simplex]
                plt.fill(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color=facecolor, alpha=fill_alpha, edgecolor=edgecolor, lw=0.3, zorder=2)

            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{filename_prefix}_{outname}.png", bbox_inches="tight", pad_inches=0)
            plt.close()

        draw_triangle_map(global_pairwise, "orange", "orangered", 0.7, "two_stromatolite")
        draw_triangle_map(global_multi, "#9B59B6", "#4B0082", 0.83, "three_stromatolite")
        draw_triangle_map(global_isolated, "royalblue", "midnightblue", 0.45, "one_stromatolite")

    return df


def target_void_split(target_id, all_points, exterior_triangles, region_info, structure_ids, img, 
                      save_dir="./results/Delaunay"):
    """Visualize void triangles around one stromatolite, split by triangle type."""

    region_dict = {reg["id"]: reg for reg in region_info}
    target_triangles = []
    neighbor_structures = set()

    for simplex in exterior_triangles:
        vertex_structs = structure_ids[simplex]

        if target_id in vertex_structs:
            target_triangles.append(simplex)

            for sid in np.unique(vertex_structs):
                if sid != target_id:
                    neighbor_structures.add(sid)

    neighbors = sorted(neighbor_structures)

    print(f"Structure {target_id} neighbors: {neighbors}")

    pairwise_tris = []
    multi_tris = []
    isolated_tris = []

    for simplex in target_triangles:
        unique_ids = np.unique(structure_ids[simplex])

        if len(unique_ids) == 2:
            pairwise_tris.append(simplex)
        elif len(unique_ids) == 3:
            multi_tris.append(simplex)
        elif len(unique_ids) == 1:
            isolated_tris.append(simplex)

    print(f"Total void triangles = {len(target_triangles)}")
    print(f"Pairwise triangles = {len(pairwise_tris)}")
    print(f"Multi triangles = {len(multi_tris)}")
    print(f"Isolated triangles = {len(isolated_tris)}")

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0, zorder=0)

    for simplex in exterior_triangles:
        pts = all_points[simplex]
        plt.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color="gray", lw=0.3, alpha=0.4, zorder=1)

    for simplex in multi_tris:
        pts = all_points[simplex]
        plt.fill(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color="#9B59B6", alpha=0.45, edgecolor="#4B0082", lw=0.4, zorder=2)

    for simplex in pairwise_tris:
        pts = all_points[simplex]
        plt.fill(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color="orange", alpha=0.45, edgecolor="orangered", lw=0.4, zorder=3)

    for simplex in isolated_tris:
        pts = all_points[simplex]
        plt.fill(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color="royalblue", alpha=0.45, edgecolor="midnightblue", lw=0.3, zorder=4)

    for sid in neighbors:
        cnt = region_dict[sid]["coords"]
        cnt_closed = np.vstack([cnt, cnt[0]])
        plt.plot(cnt_closed[:, 0], cnt_closed[:, 1], color="lime", lw=0.7, zorder=6)

    cnt = region_dict[target_id]["coords"]
    cnt_closed = np.vstack([cnt, cnt[0]])
    plt.plot(cnt_closed[:, 0], cnt_closed[:, 1], color="red", lw=0.7, zorder=7)

    plt.axis("off")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/void_structure_{target_id}.png"
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0) #, dpi=300

    plt.close()


def plot_shared_triangles(sid1, sid2, region_info, all_points, structure_ids, exterior_triangles, img, save_dir="./results/Delaunay"):
    """Visualize shared exterior triangles between two stromatolites."""

    region_dict = {reg["id"]: reg for reg in region_info}

    # Shared triangles
    pair_triangles = []
    for simplex in exterior_triangles:
        if set(np.unique(structure_ids[simplex])) == {sid1, sid2}:
            pair_triangles.append(simplex)

    # Shared area
    # pair_area_sum = sum(Polygon(all_points[simplex]).area for simplex in pair_triangles)

    # Contours
    cnt1 = region_dict[sid1]["coords"]
    cnt2 = region_dict[sid2]["coords"]

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0, zorder=0)

    # All exterior triangles
    for simplex in exterior_triangles:
        pts = all_points[simplex]
        plt.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color="gray", lw=0.3, alpha=0.4, zorder=1)

    # Contour points
    plt.scatter(all_points[:, 0], all_points[:, 1], s=0.5, color="blue", edgecolors="none", zorder=2)

    # Shared triangles
    for simplex in pair_triangles:
        pts = all_points[simplex]
        plt.fill(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color="orange", alpha=0.70, edgecolor="orangered", lw=0.5, zorder=3)

    # Two stromatolite contours
    plt.plot(cnt1[:, 0], cnt1[:, 1], color="red", lw=0.7, zorder=4)
    plt.plot(cnt2[:, 0], cnt2[:, 1], color="lime", lw=0.7, zorder=4)

    plt.axis("off")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/void_shared_{sid1}_{sid2}.png", 
                    bbox_inches="tight", pad_inches=0)#, dpi=300

    plt.close()


def plot_shared_triangles_clean(sid1, sid2, region_info, all_points, structure_ids, exterior_triangles, img, save_dir="./results/Delaunay"):
    """Plot shared two-stromatolite triangles between two structures."""

    region_dict = {reg["id"]: reg for reg in region_info}

    pair_triangles = []
    for simplex in exterior_triangles:
        if set(np.unique(structure_ids[simplex])) == {sid1, sid2}:
            pair_triangles.append(simplex)

    cnt1 = region_dict[sid1]["coords"]
    cnt2 = region_dict[sid2]["coords"]

    # Automatic display range
    all_xy = np.vstack([cnt1, cnt2] + [all_points[s] for s in pair_triangles])
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    margin = 0.1 * max(xmax - xmin, ymax - ymin)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Filled triangles
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0, zorder=0)

    for simplex in pair_triangles:
        pts = all_points[simplex]
        ax.fill(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color="orange", alpha=0.70, edgecolor="orangered", linewidth=0.5, zorder=2)

    ax.plot(cnt1[:, 0], cnt1[:, 1], color="red", lw=1.0, zorder=3)
    ax.plot(cnt2[:, 0], cnt2[:, 1], color="lime", lw=1.0, zorder=3)

    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymax + margin, ymin - margin)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/shared_triangles_{sid1}_{sid2}_fill.png", dpi=300, bbox_inches="tight", pad_inches=0)

    plt.close()

    # Edge-only triangles
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0, zorder=0)

    for simplex in pair_triangles:
        pts = all_points[simplex]
        ax.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color="orangered", lw=0.5, alpha=0.70, zorder=1)

    ax.plot(cnt1[:, 0], cnt1[:, 1], color="red", lw=1.0, zorder=2)
    ax.plot(cnt2[:, 0], cnt2[:, 1], color="lime", lw=1.0, zorder=2)

    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymax + margin, ymin - margin)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/shared_triangles_{sid1}_{sid2}_edge.png", dpi=300, bbox_inches="tight", pad_inches=0)

    plt.close()

