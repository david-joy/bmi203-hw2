from hw2 import cluster
from hw2 import io

import os

import numpy as np


def test_mscluster():

    coords = np.array([
        [0, 1],
        [1, 1],
        [10, 0],
        [10, 1],
    ])

    # Mock active sites with an index list
    active_sites = [0, 1, 2, 3]

    tree = cluster.MSCluster(coords)
    tree.cluster()

    res_sites = tree.group(active_sites)
    exp_sites = [
        [0, 1],
        [2, 3],
    ]
    assert res_sites == exp_sites


def test_multidimensional_scaling():

    similarity = np.array([
        [70, 20, 10, 15],
        [20, 60, 40, 25],
        [10, 40, 55, 75],
        [15, 25, 75, 100],
    ])

    exp_distance = cluster.convert_similarity_to_distance(similarity)
    coords = cluster.multidimensional_scaling(
        similarity, num_dims=2)

    # Reference values from scikit learn's implementation of MDS
    exp_coords = np.array([
        [0.56325076, -0.16845574],
        [0.04161582, 0.43706163],
        [-0.32409198, -0.02946705],
        [-0.2807746, -0.23913884],
    ])

    # One coordinate for each active site
    # Embedded in a 2D space
    assert coords.shape == (similarity.shape[0], 2)
    np.testing.assert_almost_equal(coords, exp_coords, decimal=4)

    # Check to see if our embedding did a decent job
    distance = np.zeros_like(similarity, dtype=np.float64)
    for i in range(similarity.shape[0]):
        c1 = coords[i, :]
        for j in range(i, similarity.shape[1]):
            c2 = coords[j, :]
            dist = np.sqrt(np.sum((c1 - c2)**2))
            distance[i, j] = dist
            distance[j, i] = dist

    # 4D to 2D means it's really hard to be perfect
    assert np.all(np.abs(distance - exp_distance) < 0.05)


def test_similarity():
    filename_a = os.path.join("data", "276.pdb")
    filename_b = os.path.join("data", "4629.pdb")
    filename_c = os.path.join("data", "10701.pdb")

    activesite_a = io.read_active_site(filename_a)
    activesite_b = io.read_active_site(filename_b)
    activesite_c = io.read_active_site(filename_c)

    # Distance metric properties
    sim_a2a = cluster.compute_similarity(activesite_a, activesite_a)

    sim_a2b = cluster.compute_similarity(activesite_a, activesite_b)
    sim_b2a = cluster.compute_similarity(activesite_b, activesite_a)

    sim_a2c = cluster.compute_similarity(activesite_a, activesite_c)
    sim_b2c = cluster.compute_similarity(activesite_b, activesite_c)

    # Similarity scores, most similar should be itself
    assert sim_a2a == 2.5350261094110822
    assert sim_a2a > sim_a2b
    assert sim_a2a > sim_a2c

    # Transitivity
    assert sim_a2b == sim_b2a


def test_compute_similarity_matrix():
    filename_a = os.path.join("data", "276.pdb")
    filename_b = os.path.join("data", "4629.pdb")
    filename_c = os.path.join("data", "10701.pdb")

    activesite_a = io.read_active_site(filename_a)
    activesite_b = io.read_active_site(filename_b)
    activesite_c = io.read_active_site(filename_c)

    active_sites = [activesite_a, activesite_b, activesite_c]

    similarity = cluster.compute_similarity_matrix(active_sites)
    exp_similarity = np.array([
        [2.5350, 8.6358e-11, 2.6558e-13],
        [8.6358e-11, 0.57013, 0.024404],
        [2.6558e-13, 0.024404, 10.237]])

    np.testing.assert_almost_equal(
        similarity, exp_similarity, decimal=3)


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
    exp_clusters = {(276, 4629), (10701,)} 
    res_clusters = cluster.cluster_by_partitioning(
        active_sites, decay=0.01)
    res_clusters = {tuple([int(c.name) for c in cluster])
                    for cluster in res_clusters}
    assert res_clusters == exp_clusters


def test_hierarchical_clustering():
    # tractable subset
    pdb_ids = [276, 1806, 3458, 3733, 4629, 10701]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb" % id)
        active_sites.append(io.read_active_site(filepath))

    ac_names = {int(ac.name): ac for ac in active_sites}

    # update this assertion
    res_clusters = cluster.cluster_hierarchically(active_sites)
    exp_clusters = [
        [ac_names[4629], ac_names[10701]],
        [
            [ac_names[3458], ac_names[3733]],
            [ac_names[276], ac_names[1806]],
        ],
    ]
    assert res_clusters == exp_clusters
