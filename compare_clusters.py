
""" Read in our partitionings and output agreement """

from hw2.io import read_clustering
from hw2.metrics import pairwise_agreement

partition = read_clustering('partition.txt')
hierarchy = read_clustering('hierarchy.txt')

print(pairwise_agreement(partition, hierarchy))
print(pairwise_agreement(hierarchy, partition))
