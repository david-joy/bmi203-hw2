import numpy as np

from .blosum62 import BLOSUM62

# Constants

# Probability of an amino acid being the catalytic residue in the active site
# Kumar, S., Kumar, N., and Gaur, R.K. (2011). Amino acid frequency
# distribution at enzymatic active site. IIOABJ 2, 23–30.
ACTIVE_SITE_FREQ = {
    "ALA": 0.0102,
    "CYS": 0.0372,
    "ASP": 0.1581,
    "GLU": 0.1358,
    "PHE": 0.0149,
    "GLY": 0.0363,
    "HIS": 0.1906,
    "ILE": 0.0047,
    "LYS": 0.0753,
    "LEU": 0.0065,
    "MET": 0.0028,
    "ASN": 0.0381,
    "PRO": 0.0019,
    "GLN": 0.0242,
    "ARG": 0.0819,
    "SER": 0.0642,
    "THR": 0.0372,
    "VAL": 0.0019,
    "TRP": 0.0140,
    "TYR": 0.0642,
}


# Functions


def compute_similarity(site_a, site_b):
    """
    Compute the similarity between two given ActiveSite instances.

    Input: two ActiveSite instances
    Output: the similarity between them (a floating point number)
    """

    # Modified CPASS active site comparison algorithm

    # Combine protein substitution probability, alpha-carbon position
    # similarity, and likelihood of a residue being the key member of the
    # active site into a single similarity score

    # Powers, R., Copeland, J.C., Germer, K., Mercier, K.A., Ramanathan, V.,
    # and Revesz, P. (2006). Comparison of protein active site structures for
    # functional annotation of proteins and drug design. Proteins: Structure,
    # Function, and Bioinformatics 65, 124–135.

    similarity = 0.0

    for residue_a in site_a.residues:
        # Coordinates of alpha carbon
        ca_a = residue_a.alpha_carbon.coords

        # Residue type
        type_a = residue_a.type

        freq_a = ACTIVE_SITE_FREQ[type_a]

        # Compare this residue with **EVERY** residue in site_b
        # exponentially weighting the contribution by distance
        for residue_b in site_b.residues:
            ca_b = residue_b.alpha_carbon.coords
            type_b = residue_b.type

            score = BLOSUM62[(type_a, type_b)]
            dist = np.sqrt(sum((a - b)**2 for a, b in zip(ca_a, ca_b)))
            freq = ACTIVE_SITE_FREQ[type_b] * freq_a

            # Within one angstrom, weigh all distances equally. This prevents
            # noisy measurements from affecting the score too much
            dist = dist - 1 if dist > 1 else 0

            # Exponentially weight the distance
            # This makes the contribution of non-close residues fall off
            # quickly
            
            # Weigh by the BLOSUM score so identical residues contribute highly
            # and distinct residues barely matter
            
            # The original paper has a term for proximity to the ligand, but
            # we don't have that information, so instead weigh by how
            # frequently this amino acid is found in an active sites
            similarity += freq * np.exp(-dist) * score

    return similarity


def compute_similarity_matrix(active_sites):
    """ Compute all pairwise similarities

    :param active_sites:
        An n-length list of the active sites
    :returns:
        An n x n numpy array of similarities
    """
    # Calculate pairwise similarity matrix
    num_sites = len(active_sites)
    similarity = np.zeros((num_sites, num_sites))
    for i in range(num_sites):
        for j in range(i, num_sites):
            sim = compute_similarity(active_sites[i], active_sites[j])
            similarity[i, j] = sim
            similarity[j, i] = sim
    return similarity


def convert_similarity_to_distance(similarity):
    """ Convert the similarity matrix to a distance matrix

    :param similarity:
        An n x n symmetric matrix where high values are CLOSER
    :returns:
        An n x n symmetric normalized distance matrix
    """
    num_sites = similarity.shape[0]
    assert similarity.shape[1] == num_sites

    # We're going to artificially constrain the diagonals to be 0.0 distance
    # because anything else doesn't really make much sense
    similarity = similarity.astype(np.float64)
    max_similarity = np.max(similarity)
    
    # Normalize the maximum similarity to 1.0
    # Distance is inverse similarity (0.0 = closest) (1.0 = farthest)
    distance = 1.0 - similarity / max_similarity

    # Force all the diagonal terms to be 0.0
    distance[np.eye(num_sites, dtype=np.bool)] = 0.0
    return distance


def multidimensional_scaling(similarity, num_dims=2):
    """ Convert a set of similarities (or distances) to a coordinate system

    Distance matricies make calculating things like average coordinates really
    painful. By embedding our similarity scores in a low dimensional space, we
    get all the advantages of a set of features from a single scoring function
    without having to worry about how that score is calculated.

    This algorithm is a common alternative to tSNE for embedding high
    dimensional data into a low dimensional space.

    Algorithm from `Multidimensional Scaling <https://en.wikipedia.org/wiki/Multidimensional_scaling>`_

    Borg, I., and Groenen, P.J.F. (2005). Modern multidimensional scaling:
    theory and applications (New York: Springer).

    :param similarity:
        The complete pairwise similarity matrix (n x n)
    :param num_dims:
        The number of output dimensions to project it onto
    :returns:
        An n x ndim list of coordinates for each element in similarity
    """

    similarity = similarity.copy()
    num_sites = similarity.shape[0]
    assert similarity.shape[1] == similarity.shape[0]
    if num_sites < num_dims:
        err = 'Cannot embed {} sites in {} dimensions'
        err = err.format(num_sites, num_dims)
        raise ValueError(err)
    assert num_sites >= num_dims

    distance = convert_similarity_to_distance(similarity)

    # Create a centering matrix to center the proximity matrix
    centering = np.eye(num_sites) - np.ones((num_sites, num_sites)) / num_sites

    # In MATLAB: B = -0.5 * J * (D.^2) * J
    # But it looks cooler in python because of the mATrix multiply
    proximity = -0.5 * (centering @ distance**2 @ centering)

    # Extract the num_dims largest eigenvalues
    evals, evects = np.linalg.eig(proximity)

    # Because LAPACK is dumb, we have to sort the eigenvalues
    # Find the indicies needed to sort the array and take the top n
    indicies = np.argpartition(-evals, num_dims)[:num_dims]

    evals = np.diag(evals[indicies])
    evects = evects[:, indicies]

    return evects @ np.sqrt(evals)


def cluster_by_partitioning(active_sites, decay=0.5, num_iters=100):
    """
    Cluster a given set of ActiveSite instances using a partitioning method.

    Input: a list of ActiveSite instances
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)
    
    :param decay:
        The damping factor for messages
        Faster decay creates fewer clusters (paper value: 0.5)

    :param num_iters:
        Number of iterations to use
        More iterations creates fewer clusters (paper value: 100)
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

    # Calculate pairwise similarity matrix
    num_sites = len(active_sites)
    similarity = compute_similarity_matrix(active_sites)

    # For an unbiased seeding, set the diagonals to the median
    similarity[np.eye(num_sites, dtype=np.bool)] = np.nan
    median_similarity = np.nanmedian(similarity.flatten())
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
    similarity = compute_similarity_matrix(active_sites)

    # Convert similarity to distance
    # Rescale so max similarity is 1, min is 0, then invert
    min_sim = np.min(similarity)
    max_sim = np.max(similarity)
    distance = 1.0 - (similarity - min_sim) / (max_sim - min_sim)


    return []
