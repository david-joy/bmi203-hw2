
import matplotlib.pyplot as plt

from .metrics import project_clustering, flatten_clusters


def plot_clustering(plot_file, clustering):
    """ Plot the clusters in a scatter """

    leaf_clustering = flatten_clusters(clustering)
    coords = project_clustering(leaf_clustering, num_dims=2)



    x = []
    y = []
    c = []

    for i, coord in enumerate(coords):
        cx, cy = coord[:, 0], coord[:, 1]
        x.extend(cx)
        y.extend(cy)
        c.extend([i for _ in range(cx.shape[0])])

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    ax0, ax1, ax2 = axes

    # Zoom in on interesting regions of MDS space
    ax0.scatter(x, y, s=50, c=c, cmap='Set1')
    ax0.set_xlim([-0.06, 0.0])
    ax0.set_ylim([0.0, 0.08])

    ax0.set_xlabel('MDS Coordinate 1')
    ax0.set_ylabel('MDS Coordinate 2')

    ax1.scatter(x, y, s=50, c=c, cmap='Set1')
    ax1.set_xlim([-0.06, -0.04])
    ax1.set_ylim([-0.52, -0.5])   

    ax1.set_xlabel('MDS Coordinate 1')
    ax1.set_ylabel('MDS Coordinate 2') 

    ax2.scatter(x, y, s=50, c=c, cmap='Set1')
    ax2.set_xlim([0.48, 0.50])
    ax2.set_ylim([-0.02, 0.0])

    ax2.set_xlabel('MDS Coordinate 1')
    ax2.set_ylabel('MDS Coordinate 2')

    ax1.set_title('Clusters in MDS Space')

    plt.tight_layout()

    fig.savefig(plot_file)
    plt.close()
