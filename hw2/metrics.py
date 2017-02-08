""" Metrics to evaluate clusterings

Based on `Cluster Analysis <https://en.wikipedia.org/wiki/Cluster_analysis#Internal_evaluation>`_
"""

# 3rd party
import numpy as np

# Our own imports
from .cluster import compute_similarity_matrix, multidimensional_scaling


# Helper functions


def unpack_nested_clusters(clustering):
    """ Extract the indicies to access the leaves of a nested tree of clusters

    :param clustering:
        A list of lists of lists of... eventually ActiveSites
    :returns:
        A generator that yields pairs of index, active_site. Each unique index
        is the set of indicies needed to access that active site.

    Ex: ``(1, 2, 0), ActiveSite('123')`` means that
        ``clustering[1][2][0] == ActiveSite('123')``
    """
    targets = [((i, ), c) for i, c in enumerate(clustering)]
    while len(targets) > 0:
        index, target = targets.pop(0)
        if not hasattr(target, '__iter__'):
            yield index, target
        else:
            targets.extend((index + (i, ), t) for i, t in enumerate(target))


def flatten_clusters(clustering):
    """ Take a nested clustering and return a flat clustering

    :param clustering:
        A list of lists of lists of... eventually ActiveSites
    :returns:
        A list of lists of ActiveSites
    """

    clusters = {}
    for index, target in unpack_nested_clusters(clustering):
        assert len(index) > 1
        clusters.setdefault(index[:-1], []).append(target)

    leaf_clustering = []
    for key in sorted(clusters):
        leaf_clustering.append(clusters[key])
    return leaf_clustering


def project_clustering(clustering, num_dims=2):
    """ Take a clustering an project it into a similarity space

    :param clustering:
        A list of lists of ActiveSites
    :param num_dims:
        The number of dimensions to project onto
    :returns:
        A list of numpy arrays with the coordinates of those ActiveSites in
        similarity space
    """
    active_sites = [c for cluster in clustering for c in cluster]
    similarity = compute_similarity_matrix(active_sites)
    coords = multidimensional_scaling(similarity, num_dims=num_dims)

    clustering_coords = []
    offset = 0
    for cluster in clustering:
        clustering_coords.append(coords[offset:offset+len(cluster)])
        offset += len(cluster)
    return clustering_coords


# Clustering Metrics


def davies_bouldin_index(clustering):
    """ Davies-Bouldin Index

    :param clustering:
        The list of lists containing the clusters
    :returns:
        The index, where a smaller value indicates a better clustering
    """

    # Project the clusters back into similarity space
    leaf_clustering = flatten_clusters(clustering)
    leaf_coords = project_clustering(leaf_clustering)

    index = 0.0

    for i, ci in enumerate(leaf_coords):

        # Find the center of the cluster and the average distance from each
        # point to the center
        center_i = np.mean(ci, axis=0)[np.newaxis, :]
        dist_i = np.mean(np.sqrt(np.sum((ci - center_i)**2, axis=1)))

        metric = []

        for j, cj in enumerate(leaf_coords):
            if i == j:
                continue
            center_j = np.mean(cj, axis=0)[np.newaxis, :]
            dist_j = np.mean(np.sqrt(np.sum((cj - center_j)**2, axis=1)))

            dist_ij = np.sqrt(np.sum((center_j - center_i)**2))

            # Larger if each individual cluster has a large diameter
            # Smaller if the pair of clusters are far apart
            metric.append((dist_i + dist_j) / dist_ij)

        # Take the worst case for all the pairs
        # This will be the largest other cluster closest to this cluster
        index += np.max(metric)
    return index


def dunn_index(clustering):
    """ Dunn Index

    :param clustering:
        The list of lists containing the clusters
    :returns:
        A list of indicies for each cluster, where a larger value indicates a
        better cluster
    """

    # Project the clusters back into similarity space
    leaf_clustering = flatten_clusters(clustering)
    leaf_coords = project_clustering(leaf_clustering)

    index = []
    for i, ci in enumerate(leaf_coords):

        # Find the center of the cluster
        center_i = np.mean(ci, axis=0)[np.newaxis, :]

        # Measure within cluster distance as farthest point from the center
        dist_i = np.max(np.sqrt(np.sum((ci - center_i)**2, axis=1)))

        between_dists = []

        for j, cj in enumerate(leaf_coords):
            if i == j:
                continue
            center_j = np.mean(cj, axis=0)[np.newaxis, :]

            # Measure between cluster distance as distance between centers
            between_dists.append(
                np.sqrt(np.sum((center_j - center_i)**2, axis=1)))

        # (min distance between clusters) / (max distance within cluster)
        dist_j = np.min(between_dists)
        index.append(dist_j / dist_i)
    return index


def silhouette_index(clustering):
    """ Silhouette Coefficient

    :param clustering:
        The list of lists containing the clusters
    :returns:
        A list of indicies for each item, where a score of 1 is perfect
        membership and a score -1 is the worst assigned membership
    """

    # Project the clusters back into similarity space
    leaf_clustering = flatten_clusters(clustering)
    leaf_coords = project_clustering(leaf_clustering)

    index = []
    for i, ci in enumerate(leaf_coords):

        # Find the center of the cluster
        center_i = np.mean(ci, axis=0)[np.newaxis, :]

        # Average dissimilarity as distance from the center
        dissim_i = np.sqrt(np.sum((ci - center_i)**2, axis=1))

        dissim_best = None
        for j, cj in enumerate(leaf_coords):
            if i == j:
                continue
            center_j = np.mean(cj, axis=0)[np.newaxis, :]
            dissim_j = np.sqrt(np.sum((ci - center_j)**2, axis=1))

            # Take whichever of the between cluster distances is smaller
            # We're trying to compare to the next best cluster this point
            # could live in
            if dissim_best is None:
                dissim_best = dissim_j
            else:
                dissim_best = np.min(np.stack(
                    [dissim_best, dissim_j], axis=1), axis=1)

        assert dissim_best.shape == dissim_i.shape

        # Work out the denominator
        dissim_largest = np.max(np.stack(
            [dissim_i, dissim_best], axis=1), axis=1)

        scores = (dissim_best - dissim_i) / dissim_largest
        index.extend(scores)

    return index
