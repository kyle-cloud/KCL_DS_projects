import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
y = iris.data
target = iris.target

samples = np.array([[6.1, 3.1, 5.5, 0.5],
            [4.9, 2.0, 5.7, 0.1],
            [4.7, 3.1, 5.4, 2.1],
            [5.9, 2.1, 3.4, 0.2],
            [7.7, 2.6, 1.5, 2.4]])

k = 5
clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
clf.fit(y, target)  

print(clf.predict(samples))