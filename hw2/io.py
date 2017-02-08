import glob
import os
from .utils import Atom, Residue, ActiveSite


def read_active_sites(dir):
    """
    Read in all of the active sites from the given directory.

    Input: directory
    Output: list of ActiveSite instances
    """
    active_sites = []
    # iterate over each .pdb file in the given directory
    for filepath in glob.iglob(os.path.join(dir, "*.pdb")):

        active_sites.append(read_active_site(filepath))

    print("Read in %d active sites" % len(active_sites))

    return active_sites


def read_active_site(filepath):
    """
    Read in a single active site given a PDB file

    Input: PDB file path
    Output: ActiveSite instance
    """
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)

    if name[1] != ".pdb":
        raise IOError("%s is not a PDB file" % filepath)

    active_site = ActiveSite(name[0])

    r_num = 0

    # open pdb file
    with open(filepath, "r") as f:
        # iterate over each line in the file
        for line in f:
            if line[0:3] != 'TER':
                # read in an atom
                atom_type = line[13:17].strip()
                x_coord = float(line[30:38])
                y_coord = float(line[38:46])
                z_coord = float(line[46:54])
                atom = Atom(atom_type)
                atom.coords = (x_coord, y_coord, z_coord)

                residue_type = line[17:20]
                residue_number = int(line[23:26])

                # make a new residue if needed
                if residue_number != r_num:
                    residue = Residue(residue_type, residue_number)
                    r_num = residue_number

                # add the atom to the residue
                residue.atoms.append(atom)

            else:  # I've reached a TER card
                active_site.residues.append(residue)

    return active_site


def read_clustering(filename):
    """ Read the clustering in

    :param filename:
        The cluster file to read
    :returns:
        A list of lists of active site numbers
    """

    clusters = []
    cluster = []

    with open(filename, 'r') as fp:
        for line in fp:
            line = line.strip()
            if line == '':
                continue
            if line.startswith('---'):
                continue

            if line.startswith('Cluster'):
                if len(cluster) > 0:
                    clusters.append(cluster)
                cluster = []
            else:
                cluster.append(int(line))
    if len(cluster) > 0:
        clusters.append(cluster)
    return clusters


def write_clustering(filename, clusters):
    """
    Write the clustered ActiveSite instances out to a file.

    Input: a filename and a clustering of ActiveSite instances
    Output: none
    """

    with open(filename, 'w') as out:
        for i in range(len(clusters)):
            out.write("\nCluster %d\n--------------\n" % i)
            for j in range(len(clusters[i])):
                out.write("%s\n" % clusters[i][j])


def write_mult_clusterings(filename, clusterings):
    """
    Write a series of clusterings of ActiveSite instances out to a file.

    Input: a filename and a list of clusterings of ActiveSite instances
    Output: none
    """
    # Support nested heirarchies because two levels isn't enough...

    # Initialize the stack and counter
    targets = clusterings
    cluster_count = -1

    with open(filename, 'w') as out:
        while len(targets) > 0:
            target = targets.pop(0)
            if all([isinstance(t, ActiveSite) for t in target]):
                cluster_count += 1
                out.write("\nCluster %d\n------------\n" % cluster_count)
                for active_site in target:
                    out.write("%s\n" % active_site)
            else:
                # Got a list, add it to the stack
                targets.extend(target)
