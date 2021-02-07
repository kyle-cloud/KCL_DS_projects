import numpy as np

x = np.array([[1,0,0],
            [1,0,1],
            [1,1,0],
            [1,1,1]]).astype(float)
t = np.array([0,0,0,1]).astype(float)
w = np.array([0.5, 1,1]).astype(float)
lr = 1

def H(y):
    if y > 0:
        return 1
    elif y == 0:
        return 0.5
    else:
        return 0

for i in range(5):
    for j in range(len(x)):
        y = H(np.matmul(w, x[j].T))
        w += lr * (t[j] - y) * x[j]
        print(y, w)