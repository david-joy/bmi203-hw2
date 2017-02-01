from hw2 import cluster
from hw2 import io

import os

import numpy as np


def test_similarity():
    filename_a = os.path.join("data", "276.pdb")
    filename_b = os.path.join("data", "4629.pdb")
    filename_c = os.path.join("data", "10701.pdb")

    activesite_a = io.read_active_site(filename_a)
    activesite_b = io.read_active_site(filename_b)
    activesite_c = io.read_active_site(filename_c)

    # Distance metric properties
    dist_a2a = cluster.compute_similarity(activesite_a, activesite_a)

    dist_a2b = cluster.compute_similarity(activesite_a, activesite_b)
    dist_b2a = cluster.compute_similarity(activesite_b, activesite_a)

    dist_a2c = cluster.compute_similarity(activesite_a, activesite_c)
    dist_b2c = cluster.compute_similarity(activesite_b, activesite_c)

    # Identity
    assert dist_a2a == 0.0

    # Transitivity
    assert dist_a2b == dist_b2a

    # Triangle inequality
    assert dist_a2c <= dist_a2b + dist_b2c

    # FIXME: Update this for a **REAL** distance metric
    assert dist_a2b == 162.0762768141123


def test_partition_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb" % id)
        active_sites.append(io.read_active_site(filepath))

    # Set the random seed for repeatable testing
    np.random.seed(1)

    # Cluster order isn't guaranteed, so compare sets
    exp_clusters = {(276, ), (4629, 10701)}
    res_clusters = cluster.cluster_by_partitioning(active_sites)
    res_clusters = {tuple([int(c.name) for c in cluster])
                    for cluster in res_clusters}
    assert res_clusters == exp_clusters


def test_hierarchical_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb" % id)
        active_sites.append(io.read_active_site(filepath))

    # update this assertion
    assert cluster.cluster_hierarchically(active_sites) == []
