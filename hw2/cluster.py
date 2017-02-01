# from .utils import Atom, Residue, ActiveSite

import numpy as np


def compute_similarity(site_a, site_b):
    """
    Compute the similarity between two given ActiveSite instances.

    Input: two ActiveSite instances
    Output: the similarity between them (a floating point number)
    """

    similarity = 0.0

    # Trivial algorithm, distance between alpha carbons
    # FIXME: Align residues before comparing
    # FIXME: Do something more sophisticated!!
    for residue_a, residue_b in zip(site_a.residues, site_b.residues):
        ca_a = residue_a.alpha_carbon.coords
        ca_b = residue_b.alpha_carbon.coords

        similarity += np.sqrt(sum((a - b)**2 for a, b in zip(ca_a, ca_b)))

    # Fill in your code here!

    return similarity


def cluster_by_partitioning(active_sites):
    """
    Cluster a given set of ActiveSite instances using a partitioning method.

    Input: a list of ActiveSite instances
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)
    """

    # Affinity propagation
    # Frey, B.J., and Dueck, D. (2007). Clustering by passing messages between
    # data points. Science 315, 972–976.
    # http://www.psi.toronto.edu/affinitypropagation/FreyDueckScience07.pdf

    # Affinity propagation assigns clusters by attempting to identify
    # exemplars within the data that represent a large number of data points.
    # Each point is assigned "responsibility" for it's neighbors in proprotion
    # to how representative it is, then each point is assigned "availability"
    # based on how near to a cluster border it is. After several iterations,
    # the availability of a point decreases and it can be assigned to a single
    # cluster.

    # This is a naive implementation based on the author's MATLAB sample code
    # with O(N**3) runtime. An improved implementation would reuse operations
    # and have O(N**2) behavior on a dense similarity matrix and O(N) on a
    # sparse similarity matrix.

    # The key advantage of affinity propagation is that it does not require
    # selection of a cluster size, only a damping term that determines how
    # quickly messages decay.

    # Parameters

    # Damping factor
    # Faster decay creates fewer clusters (paper value: 0.5)
    decay = 0.1

    # Number of iterations to use
    # More iterations creates fewer clusters (paper value: 100)
    num_iters = 100

    # Calculate pairwise distance matrix
    num_sites = len(active_sites)
    similarity = np.empty((num_sites, num_sites))
    measured_sims = []
    for i in range(num_sites):
        for j in range(i+1, num_sites):
            sim = compute_similarity(active_sites[i], active_sites[j])
            measured_sims.append(sim)
            similarity[i, j] = sim
            similarity[j, i] = sim

    # For an unbiased seeding, set the diagonals to the median
    median_similarity = np.median(measured_sims)
    for k in range(num_sites):
        similarity[k, k] = median_similarity

    # Initialize message passing matrices
    responsibility = np.zeros((num_sites, num_sites))
    availability = np.zeros((num_sites, num_sites))

    # Add some noise in case similarity matrix is degenerate
    sim_range = np.max(similarity) - np.min(similarity)
    similarity += 1e-12 * sim_range * np.random.rand(num_sites, num_sites)

    for step in range(num_iters):
        # Compute responsibilities
        resp_old = responsibility.copy()
        avail_sim = availability + similarity

        # Blank out the top link
        as_max = np.max(avail_sim, axis=1)[:, np.newaxis]
        as_maxinds = np.argmax(avail_sim, axis=1)
        for i in range(num_sites):
            avail_sim[i, as_maxinds[i]] = np.nan

        # Now find the second best link
        as_max2 = np.nanmax(avail_sim, axis=1)[:, np.newaxis]

        # Downgrade responsibility along all links by strength of the best path
        responsibility = similarity - as_max

        # Except along the best path, which we downgrade by the second best
        # strength
        for i in range(num_sites):
            j = as_maxinds[i]
            responsibility[i, j] = similarity[i, j] - as_max2[i]

        # Dampen responsibility by our decay weight
        responsibility = (1 - decay) * responsibility + decay * resp_old

        # Compute availability
        avail_old = availability.copy()
        resp_prop = responsibility.copy()
        resp_prop[resp_prop < 0] = 0
        for k in range(num_sites):
            resp_prop[k, k] = responsibility[k, k]

        # New availability is the inverse of how responsible this node is
        # for every other node
        availability = np.sum(resp_prop, axis=0)[np.newaxis, :] - resp_prop

        # Force availability < 0 everywhere except the diagonal
        diag_avail = np.diag(availability.copy())
        availability[availability > 0] = 0
        for k in range(num_sites):
            availability[k, k] = diag_avail[k]

        # Dampen availability by our decay weight
        availability = (1 - decay) * availability + decay * avail_old

    # Now work out which data points are our exemplars
    # Exemplars are points with high responisibility and low availability
    exemplars = responsibility + availability
    exemplar_indicies = np.where(np.diag(exemplars) > 0)[0]
    num_clusters = exemplar_indicies.shape[0]

    # Find the closest exemplar to each point
    clusters = np.argmax(similarity[:, exemplar_indicies], axis=1)
    clusters[exemplar_indicies] = np.arange(num_clusters)

    # Group output active sites by cluster
    final_clusters = {}
    for active_site, cluster_index in zip(active_sites, clusters):
        final_clusters.setdefault(cluster_index, []).append(active_site)
    return [v for v in final_clusters.values()]


def cluster_hierarchically(active_sites):
    """
    Cluster a set of ActiveSite instances using a hierarchical algorithm.

    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of Sequence objects)
    """

    # Fill in your code here!

    return []
