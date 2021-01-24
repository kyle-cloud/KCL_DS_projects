import numpy as np
from sklearn import datasets

# load the dataset
iris = datasets.load_iris()

# create y and target
y = np.ones(len(iris.data))
y = np.insert(iris.data, 0, values=y, axis=1)

target = iris.target
target[target != 0] = -1
target[target == 0] = 1
print(target.shape)
y = y * target.reshape((-1, 1))

# parameters
a = np.array([0.5, 0.5, -1.5, 2.5, -0.5]).reshape((1, -1))
b = np.ones(len(y))

# predict before learning
prediction = np.matmul(y, a.T)
prediction[prediction > 0] = 1
prediction[prediction <= 0] = -1
print(sum(prediction == target) / len(y))

lr = 0.01

for i in range(2):
    for j in range(len(y)):
        a += lr * (b[j] - np.matmul(y[j], a.T)) * y[j]
print(a)

prediction = np.matmul(y, a.T)
prediction[prediction > 0] = 1
prediction[prediction <= 0] = -1
print(prediction)
print(sum(prediction == target) / len(y))