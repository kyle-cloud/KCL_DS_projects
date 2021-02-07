import matplotlib.pyplot as plt
import numpy as np
import random
import sklearn.cluster as cluster
import sklearn.metrics as metrics

# create the dataset
X, clusters = data.make_blobs(n_samples=1000, n_features=2, cluster_std=1.0, random_state=2021)

K = 9
km = cluster.KMeans(n_clusters=K)

km.fit(X)

SC = metrics.silhouette_score(X, km.labels_, metric='euclidean')
CH = metrics.calinski_harabasz_score(X, km.labels_)

def plot_scatter(X, clusters):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=clusters)
    fig.show()
plot_scatter(X, km.labels_)