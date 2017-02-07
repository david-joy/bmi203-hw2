""" Metrics to evaluate clusterings

Based on `Cluster Analysis <https://en.wikipedia.org/wiki/Cluster_analysis#Internal_evaluation>`_
"""


# Our own imports
from .cluster import compute_similarity

# Clustering Metrics


def davies_bouldin_index(clustering):
    """ Davies-Bouldin Index

    :param clustering:
        The list of lists containing the clusters
    :returns:
        The index, where a smaller value indicates a better clustering
    """


    return 0


def dunn_index(clustering):
    """ Dunn Index

    :param clustering:
        The list of lists containing the clusters
    :returns:
        A list of indicies for each cluster, where a larger value indicates a better cluster
    """
    index = []
    for cluster in clustering:
        distances = []
        for site1, site2 in itertools.combinations(cluster, 2):
            distances.append(compute_similarity(site1, site2))
        min_distance = min(distances)
        max_distance = max(distances)
        index.append(min_distance/max_distance)



def silhouette_index(clustering):
    """ Silhouette Coefficient

    :param clustering:
        The list of lists containing the clusters
    :returns:
        A list of indicies for each cluster, where a score of 1 is perfect clustering and a score -1 is the worst clustering
    """
    
