# -*- coding: utf-8 -*-
"""
@author: zhouziyun
"""

# main.py should be placed one level above the stromanet package.
# Add the project root directory to sys.path if necessary.

from src import preprocessing, image_segmentation, delaunay, network, metrics, visualization


# ------------------------------
# image preprocessing
# ------------------------------
# load image
img = preprocessing.read_image(img_path="image/stromatolite1.png")

# px m transfer
meter_per_px = preprocessing.get_meter_per_pixel(img, real_length_m=2)

# convert to grayscale image
img_gray = preprocessing.grayscale(img)

# Gaussian blur
img_blur = preprocessing.Gaussian_blur(img_gray)

# binarization
img_binary = preprocessing.Otsu(img_blur, black_thresh=45)

# morphological processing
closed, sure_bg = preprocessing.morphological_cleanup(img_binary, iteration_open=1, iteration_close=1)

# distance transform to find sure foreground
sure_fg, unknown = preprocessing.extract_foreground_unknown(closed, sure_bg, threshold_ratio=0.26)

# plot background, sure foreground and unknown region
preprocessing.plot_watershed_regions(sure_bg, sure_fg, unknown)


# ------------------------------
# segmentation
# ------------------------------
# Watershed 
markers = image_segmentation.watershed(img, sure_fg, unknown)

# Contour extraction
region_info = image_segmentation.extract_contours_cleancolor(img, markers, img_alpha=0, stroma_alpha=1)

# # Generate shapefile for use in GAMA
# gdf = image_segmentation.gama_shapefile(img, region_info)


# ------------------------------
# Delaunay triangulation using all stromatolites contour points
# ------------------------------
all_points, structure_ids, point_meta = delaunay.collect_delaunay_points(region_info)

tri = delaunay.compute_delaunay(all_points, img, bg_alpha=0)

# exterior triangles
exterior_triangles = delaunay.filter_exterior_triangles(tri, all_points, region_info, img, bg_alpha=0)

# plot triangles edge length distribution
edge_lengths_all, edge_lengths_ext, filtered_lengths_all, filtered_lengths_ext = delaunay.plot_edge_length_distribution(
    all_points, tri, exterior_triangles, point_meta, show_stats=False)


# ------------------------------
# void
# ------------------------------
# compute void metrics(3 group)
df_void = delaunay.compute_void_metrics(region_info, all_points, exterior_triangles, structure_ids, img)

# show an example structure with its neighbors
target_idx = 11
delaunay.target_void_split(target_idx, all_points, exterior_triangles, region_info, structure_ids, img)

# show a pair of example interaction structures
delaunay.plot_shared_triangles(11, 34, region_info, all_points, structure_ids, exterior_triangles, img)
# zoom in
delaunay.plot_shared_triangles_clean(11, 34, region_info, all_points, structure_ids, exterior_triangles, img)


# ------------------------------
# Graph definition1 (triangle-area network: G_tri)
# ------------------------------
# bulid network
G_tri, edge_w_tri = network.build_void_triangle_network(region_info, all_points, exterior_triangles, structure_ids)

# calculate and plot network metrics
metrics_df = metrics.analyse_network_metrics(G_tri, region_info, img, weight_type="proximity")

# relationship between metrics ("kA", "sA", "kP", "sP", "sk", "sA_kA", "sP_kP")
# Fit power-law scaling relationships
metrics.metric_scalings(metrics_df, weight_type="proximity")

# degree vs strength grouped by area
area_groups, area_bounds = visualization.group_by_area(dict(zip(metrics_df["id"], metrics_df["area"])), n_groups=5)
visualization.plot_area_groups_spatial(area_groups, region_info, img, n_groups=5,
    save_path="./results/network/proximity/area_groups_spatial.png")
visualization.plot_group_scaling(dict(zip(metrics_df["id"], metrics_df["degree"])),
    dict(zip(metrics_df["id"], metrics_df["strength"])), 
    area_groups, xlabel="$k$", ylabel="$s$",
    save_path="./results/network/proximity/area_groups_scaling.png")

# NW
nw = metrics.Nw(G_tri)
visualization.plot_metric_distribution(list(nw.values()), xlabel="$Nw$", bins=60, save_path="./results/network/proximity/Nw_distribution.png")
visualization.plot_metric_spatial(G_tri, nw, region_info, img, label="Nw", color_style="rgb", save_path="./results/network/proximity/Nw_spatial.png")


# ------------------------------
# Build graph definition2 (Edge count network: G_edgecount)
# ------------------------------
G_edgecount, w_edgecount = network.build_shared_edgecount_network(region_info, all_points, exterior_triangles, structure_ids)

# calculate and plot network metrics
metrics_df = metrics.analyse_network_metrics(G_edgecount, region_info, img, weight_type="obstruction")

# relationship between metrics ("kA", "sA", "kP", "sP", "sk", "sA_kA", "sP_kP")
# Fit power-law scaling relationships
metrics.metric_scalings(metrics_df, weight_type="obstruction")

# degree vs strength grouped by area
area_groups, area_bounds = visualization.group_by_area(dict(zip(metrics_df["id"], metrics_df["area"])), n_groups=5)
visualization.plot_area_groups_spatial(area_groups, region_info, img, n_groups=5,
    save_path="./results/network/obstruction/area_groups_spatial.png")
visualization.plot_group_scaling(dict(zip(metrics_df["id"], metrics_df["degree"])),
    dict(zip(metrics_df["id"], metrics_df["strength"])), 
    area_groups, xlabel="$k$", ylabel="$s$",
    save_path="./results/network/obstruction/area_groups_scaling.png")

# NW
nw = metrics.Nw(G_edgecount)
visualization.plot_metric_distribution(list(nw.values()), xlabel="$Nw$", bins=60, save_path="./results/network/obstruction/Nw_distribution.png")
visualization.plot_metric_spatial(G_edgecount, nw, region_info, img, label="Nw", color_style="rgb", save_path="./results/network/obstruction/Nw_spatial.png")



