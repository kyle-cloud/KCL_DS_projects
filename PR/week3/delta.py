import numpy as np

W = np.array([[1,1,0],
    [1,1,1]]).astype(float)
x = np.array([1,1,0]).astype(float)
alpha = 0.25
y = np.array([0,0]).astype(float)

for i in range(2):
    for j in range(len(y)):
        print(np.matmul(a, y[j].T))
        a += lr * (b[j] - np.matmul(a, y[j].T)) * y[j]
        print(a)