import sys
from .io import read_active_sites, write_clustering, write_mult_clusterings
from .cluster import cluster_by_partitioning, cluster_hierarchically
from .plots import plot_clustering
from .metrics import summarize_clustering

# Some quick stuff to make sure the program is called correctly
if len(sys.argv) < 4:
    print("Usage: python -m hw2 [-P| -H] <pdb directory> <output file>")
    sys.exit(0)

active_sites = read_active_sites(sys.argv[2])

cluster_file = sys.argv[3]
plot_file = 'images/' + cluster_file.rsplit('.', 1)[0] + '.png'

# Choose clustering algorithm
if sys.argv[1][0:2] == '-P':
    print("Clustering using Partitioning method")
    clustering = cluster_by_partitioning(active_sites)

    summarize_clustering(clustering)

    plot_clustering(plot_file, clustering)
    write_clustering(cluster_file, clustering)

if sys.argv[1][0:2] == '-H':
    print("Clustering using hierarchical method")
    clusterings = cluster_hierarchically(active_sites)

    summarize_clustering(clusterings)

    plot_clustering(plot_file, clusterings)
    write_mult_clusterings(cluster_file, clusterings)
