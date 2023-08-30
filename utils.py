import numpy as np
from numpy.random import default_rng

def gen_views(cluster_scheme, n_rows=10):
    """cluster scheme takes the form of: [(n_clusters, n_columns), (n_clusters, n_colums)]"""
    views = []
    for n_clusters, n_cols in cluster_scheme:
        views.append(gen_view(n_rows, n_clusters, n_cols)) 

    return np.concatenate(views, axis=1)

def gen_view(n_rows, n_clusters, n_cols, spacing=3, scale=.3):
    rng = default_rng()

    cluster_lengths = [n_rows // n_clusters for i in range(n_clusters)]
    cluster_lengths[0] += n_rows % n_clusters

    clusters = []

    for i, cluster_len in enumerate(cluster_lengths):
        clusters.append(rng.normal(loc=spacing*i, scale=scale, size=(n_cols, cluster_len)))

    return np.concatenate(clusters, axis=1).T