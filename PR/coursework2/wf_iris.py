import numpy as np
from sklearn import datasets

# load the dataset

# iris = datasets.load_iris()

# # create y and target
# y = np.ones(len(iris.data))
# y = np.insert(iris.data, 0, values=y, axis=1)

# target = iris.target
# target[target != 0] = -1
# target[target == 0] = 1
# print(target.shape)
# y = y * target.reshape((-1, 1))

import scipy.io

mat = scipy.io.loadmat('C:\\Users\\11394\\Documents\\GitHub\\KCL_DS_projects\\PR\\coursework2\\iris_class1_2_3_4D.mat') 
# create y and target
y = np.ones(len(mat['X'].T))
y = np.insert(mat['X'].T, 0, values=y, axis=1)

target = np.squeeze(mat['t']).astype(float)
target[target != 0] = -1
target[target == 0] = 1
print(target.shape)
y = y * target.reshape((-1, 1))




# parameters
a = np.array([0.5, 0.5, -1.5, 2.5, -0.5])
b = np.ones(len(y))

# predict before learning
y_origin = y * target.reshape((-1, 1))
prediction = np.matmul(y_origin, a.T)
prediction[prediction > 0] = 1
prediction[prediction <= 0] = -1
print(prediction)
print(sum(prediction == target) / len(y))

lr = 0.01

for i in range(2):
    for j in range(len(y)):
        a += lr * (b[j] - np.matmul(a, y[j].T)) * y[j]
print(a)

y_origin = y * target.reshape((-1, 1))
prediction = np.matmul(y_origin, a.T)
prediction[prediction > 0] = 1
prediction[prediction <= 0] = -1
print(prediction)
print(sum(prediction == target) / len(y))