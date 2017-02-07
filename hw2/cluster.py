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

# Classes


class MSCluster(object):
    """ Macnaughton-Smith Clustering tree

    Store a mapping of coordinates to original indicies in the active site
    array. This way I don't get headaches from a bunch of nested lists.
    
    It preforms a 2-way split at each level

    :param coords:
        An n x k list of coordinates, n > k
    :param index:
        An n x 1 list of indicies mapping coords to active_sites
        If None, coords is assumed to be the entire array
    """

    def __init__(self, coords, index=None):
        self.coords = coords
        if index is None:
            index = np.arange(self.coords.shape[0])
        self.index = index  # Index in the **original** active_site array

        self.children = []

    def cluster(self):
        """ Recursively split all of coords into child nodes """

        nodes = [self]
        while len(nodes) > 0:
            node = nodes.pop(0)
            node.cluster_once()
            nodes.extend(node.children)

    def cluster_once(self):
        """ Work out the splits for this node """

        # Base case - Can't split a node with two or fewer members
        if self.coords.shape[0] <= 2:
            self.children = []
            return

        # Find the farthest node from the center
        cluster_center = np.mean(self.coords, axis=0)[np.newaxis, :]
        cluster_dist = np.sum((self.coords - cluster_center)**2, axis=1)
        
        # Add the object farthest from the center to the splinter group
        splinter_indicies = [int(np.argmax(cluster_dist))]

        while True:
            remainder_indicies = [i for i in range(self.coords.shape[0])
                                  if i not in splinter_indicies]
            splinter_coords = self.coords[splinter_indicies, :]
            remainder_coords = self.coords[remainder_indicies, :]

            if remainder_coords.shape[0] < 2:
                # Everything splintered off (happens if we try to split a 2 point node)
                remainder_indicies = [i for i in range(self.coords.shape[0])]
                splinter_indicies = []
                break

            # Try to find a point nearer to the splinter group than the remainder group
            diffs = self.calc_splinter_diffs(remainder_coords, splinter_coords)

            if np.all(diffs <= 0):
                # Splinter group failed, merge back into the fold
                if len(splinter_indicies) <= 1:
                    remainder_indicies = [i for i in range(self.coords.shape[0])]
                    splinter_indicies = []
                break

            # Add the least happy element to the splinter group
            next_splinter = remainder_indicies[np.argmax(diffs)]
            splinter_indicies.append(next_splinter)

        # Now, if we got any splits, make a tree
        if len(splinter_indicies) > 0 and len(remainder_indicies) > 0:
            splinter = MSCluster(coords=self.coords[splinter_indicies, :],
                                 index=self.index[splinter_indicies])
            remainder = MSCluster(coords=self.coords[remainder_indicies, :],
                                  index=self.index[remainder_indicies])
            self.children = [splinter, remainder]
        else:
            self.children = []

    def calc_splinter_diffs(self, remainder_coords, splinter_coords):
        """ Calculate the splinter set differences

        Diff(X) = mean(dist(X, A)) - mean(dist(X, B))

        where A is the remainder set and B is the splinter set.

        :param remainder_coords:
            The n x k coords of the objects in the remainder set
        :param splinter_coords:
            The m x k coords of the objects in the splinter set
        :returns:
            An n x 1 array of differences between average distances for each
            object in the remainder set
        """

        diffs = []
        for i, coord in enumerate(remainder_coords):

            diff_a = []
            for j, rc in enumerate(remainder_coords):
                # Not really fair to consider distance to itself
                if i == j:
                    continue
                diff_a.append(np.sum((coord - rc)**2))
            diff_a = np.mean(diff_a)

            diff_b = []
            for sc in splinter_coords:
                diff_b.append(np.sum((coord - sc)**2))

            diff_b = np.mean(diff_b)
            diffs.append(diff_a - diff_b)
        return np.array(diffs)

    def group(self, data):
        """ Using the splits in cluster, return a grouping of data

        :param data:
            A list of objects with the same length as coords
        :returns:
            A nested list of the objects clustered
        """
        # Base case
        if self.children == []:
            return [data[i] for i in self.index]

        # Force the childeren to group the data
        grouping = []
        for child in self.children:
            grouping.append(child.group(data))
        return grouping


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


def cluster_hierarchically(active_sites, num_dims=2):
    """
    Cluster a set of ActiveSite instances using a hierarchical algorithm.

    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of Sequence objects)
    """
    
    # Macnaughton-Smith Divisive Clustering
    # Macnaughton-Smith, P., Williams, W., Dale, M., and Mockett, L. (1964).
    # Dissimilarity analysis: a new technique of hierarchical sub-division.
    # Nature 202, 1034–1035.

    # Seriously though, it's pretty crazy that you can do this O(N**2)
    # Although the hidden eigenvalue calculation in MDS makes this O(N**3)
    # There's probably a more clever way to do this, but it's hard with a
    # general non-linear metric.

    # Project the active site similarity into a metric space
    similarity = compute_similarity_matrix(active_sites)
    coords = multidimensional_scaling(similarity, num_dims=num_dims)

    # Use a tree to run the MS Clustering algorithm
    tree = MSCluster(coords)
    tree.cluster()
    return tree.group(active_sites)
