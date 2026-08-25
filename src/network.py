# -*- coding: utf-8 -*-
"""
@author: zhouziyun
"""


import networkx as nx
import numpy as np
from shapely.geometry import Polygon


def build_void_triangle_network(region_info, all_points, exterior_triangles, structure_ids):
    """Build a weighted stromatolite network from two-stromatolite exterior triangles."""

    G_tri = nx.Graph()

    # Add nodes
    for reg in region_info:
        sid = reg["id"]
        cx, cy = reg["centroid"]
        G_tri.add_node(sid, pos=(cx, cy), area=reg["area"])

    # Accumulate shared triangle areas
    edge_weights = {}

    for simplex in exterior_triangles:
        unique_ids = np.unique(structure_ids[simplex])

        if len(unique_ids) == 2:
            sid1, sid2 = sorted(unique_ids)
            tri_area = Polygon(all_points[simplex]).area
            edge_weights[(sid1, sid2)] = edge_weights.get((sid1, sid2), 0) + tri_area

    # Add weighted edges
    for (sid1, sid2), weight in edge_weights.items():
        G_tri.add_edge(sid1, sid2, weight=weight)

    return G_tri, edge_weights


def build_shared_edgecount_network(region_info, all_points, exterior_triangles, structure_ids):
    """
    Build stromatolite network where edge weight = number of shared Delaunay edges
    between two structures.
    """

    G_edgecount = nx.Graph()

    # Add nodes
    for reg in region_info:
        sid = reg["id"]
        cx, cy = reg["centroid"]
        G_edgecount.add_node(sid, pos=(cx, cy), area=reg["area"])

    edge_weights = {}

    # Loop over exterior triangles
    for simplex in exterior_triangles:

        vertex_structs = structure_ids[simplex]
        unique_ids = np.unique(vertex_structs)

        # Only process triangles touching exactly two structures
        if len(unique_ids) == 2:
            sid1, sid2 = sorted(unique_ids)

            tri_edge_count = 0
            edges_idx = [(0, 1), (1, 2), (2, 0)]

            for a, b in edges_idx:
                if (vertex_structs[a] == sid1 and vertex_structs[b] == sid2) or \
                   (vertex_structs[a] == sid2 and vertex_structs[b] == sid1):
                    tri_edge_count += 1

            edge_weights[(sid1, sid2)] = edge_weights.get((sid1, sid2), 0) + tri_edge_count

    # Add edges to network
    for (sid1, sid2), w in edge_weights.items():
        G_edgecount.add_edge(sid1, sid2, weight=w)

    return G_edgecount, edge_weights

