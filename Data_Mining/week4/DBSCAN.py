import matplotlib.pyplot as plt
import numpy as np
import random
import sklearn.cluster as cluster
import sklearn.neighbors as neighbors
import sklearn.metrics as metrics

# create the dataset
X, clusters = data.make_blobs(n_samples=1000, n_features=2, cluster_std=1.0, random_state=2021)

db = cluster.DBSCAN(eps=0.5, min_samples=1)
db.fit(X)


nn = neighbors.NearestNeighbors(n_neighbors=2, metric='euclidean')
nn.fit(X)

dist, ind = nn.kneighbors(X, n_neighbors=2)