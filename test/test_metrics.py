
import os

import numpy as np

from hw2 import metrics
from hw2 import io

# Helpers


def _load_active_site(idx):
    # Load the active site data for a given PDB id
    filepath = os.path.join("data", "%i.pdb" % idx)
    return io.read_active_site(filepath)


# Tests


def test_unpack_nested_clusters():
    # Make sure we can unpack nested trees of clusters
    # It should return (indicies into list, value) for each leaf in the list
    simple_clusters = [
        [3458, 3733, 4629, 10701],
        [276, 1806],
    ]

    exp_clusters = [
        ((0, 0), 3458),
        ((0, 1), 3733),
        ((0, 2), 4629),
        ((0, 3), 10701),
        ((1, 0), 276),
        ((1, 1), 1806),
    ]
    res_clusters = list(metrics.unpack_nested_clusters(simple_clusters))
    assert res_clusters == exp_clusters

    nested_clusters = [
        [4629, 10701],
        [
            [3458, 3733],
            [276, 1806],
        ],
    ]
    exp_clusters = [
        ((0, 0), 4629),
        ((0, 1), 10701),
        ((1, 0, 0), 3458),
        ((1, 0, 1), 3733),
        ((1, 1, 0), 276),
        ((1, 1, 1), 1806),
    ]
    res_clusters = list(metrics.unpack_nested_clusters(nested_clusters))
    assert res_clusters == exp_clusters


def test_flatten_clusters():
    # Make sure we can flatten clusters so we can evaluate all algorithms
    # like they're partitioning algorithms

    # Make sure we don't scramble already flat clusters
    simple_clusters = [
        [3458, 3733, 4629, 10701],
        [276, 1806],
    ]
    exp_clusters = [
        [3458, 3733, 4629, 10701],
        [276, 1806],
    ]
    res_clusters = metrics.flatten_clusters(simple_clusters)
    assert res_clusters == exp_clusters

    # And that we can unpack nested clusters
    nested_clusters = [
        [4629, 10701],
        [
            [3458, 3733],
            [276, 1806],
        ],
    ]
    exp_clusters = [
        [4629, 10701],
        [3458, 3733],
        [276, 1806],
    ]
    res_clusters = metrics.flatten_clusters(nested_clusters)
    assert res_clusters == exp_clusters


def test_project_clustering():
    # Project a clustering into our similarity space

    simple_clusters = [
        [_load_active_site(idx) for idx in [3458, 3733, 4629, 10701]],
        [_load_active_site(idx) for idx in [276, 1806]],
    ]
    exp_coords = [
        np.array([
            [-0.4263735, 0.0121979],
            [-0.4265745, 0.0128635],
            [0.233523, -0.3735359],
            [0.1690443, -0.3463151],
        ]),
        np.array([
            [0.2251903,  0.3473948],
            [0.2251903,  0.3473948],
        ]),
    ]
    res_coords = metrics.project_clustering(simple_clusters)
    assert len(res_coords) == len(exp_coords)

    for res, exp in zip(res_coords, exp_coords):
        np.testing.assert_almost_equal(res, exp)


def test_davies_bouldin_index():
    # It likes small clusters, far apart from each other (smaller is better)

    # This means that it likes lots of splits, which the nested clusters do
    # more of than the partitioning algorithm does
    simple_clusters = [
        [_load_active_site(idx) for idx in [3458, 3733, 4629, 10701]],
        [_load_active_site(idx) for idx in [276, 1806]],
    ]
    assert metrics.davies_bouldin_index(simple_clusters) == 1.175474302839556

    nested_clusters = [
        [_load_active_site(idx) for idx in [4629, 10701]],
        [
            [_load_active_site(idx) for idx in [3458, 3733]],
            [_load_active_site(idx) for idx in [276, 1806]],
        ],
    ]
    assert metrics.davies_bouldin_index(nested_clusters) == 0.1473113658623218


def test_dunn_index():
    # It likes dense clusters, far apart from each other (larger is better)

    # This means that it likes lots of splits, which the nested clusters do
    # more of than the partitioning algorithm does

    # It REAAAAALY likes 276 and 1806, which are duplicate active sites after
    # projection
    simple_clusters = [
        [_load_active_site(idx) for idx in [3458, 3733, 4629, 10701]],
        [_load_active_site(idx) for idx in [276, 1806]],
    ]
    exp_scores = [1.5537862881596853, 462826056.40496564]
    res_scores = metrics.dunn_index(simple_clusters)
    assert res_scores == exp_scores

    nested_clusters = [
        [_load_active_site(idx) for idx in [4629, 10701]],
        [
            [_load_active_site(idx) for idx in [3458, 3733]],
            [_load_active_site(idx) for idx in [276, 1806]],
        ],
    ]
    exp_scores = [20.223857599219727, 2099.9151313878456, 527446233.30976832]
    res_scores = metrics.dunn_index(nested_clusters)
    assert res_scores == exp_scores


def test_silhouette_index():
    # It likes dense clusters with minimal overlap (-1 is worst, 1 is best)

    # This means it likes very tiny clusters because they're all really close
    # in similarity space. It also finds the duplicates 276, 1806, which is
    # a good sign.

    simple_clusters = [
        [_load_active_site(idx) for idx in [3458, 3733, 4629, 10701]],
        [_load_active_site(idx) for idx in [276, 1806]],
    ]
    exp_scores = [0.5022, 0.5014, 0.4456, 0.5253, 1.0, 1.0]
    res_scores = metrics.silhouette_index(simple_clusters)
    assert np.allclose(res_scores, exp_scores, atol=1e-2)

    nested_clusters = [
        [_load_active_site(idx) for idx in [4629, 10701]],
        [
            [_load_active_site(idx) for idx in [3458, 3733]],
            [_load_active_site(idx) for idx in [276, 1806]],
        ],
    ]
    exp_scores = [0.9514, 0.9496, 0.9995, 0.9995, 1.0, 1.0]
    res_scores = metrics.silhouette_index(nested_clusters)
    assert np.allclose(res_scores, exp_scores, atol=1e-2)
