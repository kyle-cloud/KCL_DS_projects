import sklearn.datasets as data
import matplotlib.pyplot as plt
import numpy as np
import random

# create the dataset
X, clusters = data.make_blobs(n_samples=100, n_features=2, cluster_std=1.0, random_state=2021)

def plot_scatter(X, clusters):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=clusters)
    fig.show()

def distance(a, b):
    return np.sqrt(np.matmul(a - b, (a - b).T))

def WC(X, y, centres):
    res = 0
    for i in range(len(X)):
        res += distance(X[i], centres[y[i]])
    return res

def BC(centres):
    res = 0
    for i in range(len(centres)):
        for j in range(len(centres)):
            if i != j:
                res += distance(centres[i], centres[j])
    return res

## initial K centres
K = 9
min_0, max_0 = min(X[:, 0]), max(X[:, 0])
min_1, max_1 = min(X[:, 1]), max(X[:, 1])
# centres = np.array([[random.uniform(0, max_0), random.uniform(0, max_1)] for _ in range(K)])
centres = np.array([X[random.randint(0, len(X)-1)] for _ in range(K)])

## loop
y = np.array([0] * len(X))
converged = False
plot_scatter(X, y)
while not converged:
    converged = True
    # cluster
    for i in range(len(X)):
        cur_c = y[i]
        # cluster this point
        closest_d = float('inf')
        for j in range(K):
            dist = distance(X[i], centres[j])
            if dist < closest_d:
                y[i], closest_d = j, dist
        # judge if the point has changed    
        if y[i] != cur_c:
            converged = False
    plot_scatter(X, y)
    
    # update centres
    for i in range(K):
        if X[y == i].any():
            centres[i][:] = np.mean(X[y == i], axis=0)

wc = WC(X, y, centres)
bc = BC(centres)
score = bc / wc
print("WC=%f, BC=%f, score=%f" % (wc, bc, bc/wc))