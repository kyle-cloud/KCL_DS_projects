import numpy as np
y = [[1, 0, 2], 
    [1, 1, 2],
    [1, 2, 1],
    [-1, 3, -1],
    [-1, 2, 1],
    [-1, 3, 2]]
a = [1, 0, 0]
b = [1, 0.5, 0.5, 0.5, 2.5, 1]

y = np.array(y).astype(np.float)
a = np.array(a).astype(np.float)
b = np.array(b).astype(np.float)
lr = 0.1

for i in range(2):
    for j in range(len(y)):
        print(np.matmul(a, y[j].T))
        a += lr * (b[j] - np.matmul(a, y[j].T)) * y[j]
        print(a)