import matplotlib.pyplot as plt
import numpy as np
import random
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import scipy.cluster.hierarchy as hierarchy

# create the dataset
X, clusters = data.make_blobs(n_samples=1000, n_features=2, cluster_std=1.0, random_state=2021)

K = 9
hc = cluster.AgglomerativeClustering(n_clusters=K, linkage='average', affinity='euclidean')

hc.fit(X)

plt.figure()
Z = np.empty([len(hc.children_), 4], dtype=float)
cluster_distance = np.arange(hc.children_.shape[0])
cluster_sizes = np.arange(2, hc.children_.shape[0] + 2)
for i in range(len(hc.children_)):
    Z[i][0] = hc.children_[i][0]
    Z[i][1] = hc.children_[i][1]
    Z[i][2] = cluster_distance[i]
    Z[i][3] = cluster_sizes[i]
hierarchy.dendrogram(Z)
plt.show()