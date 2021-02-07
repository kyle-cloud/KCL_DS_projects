import numpy as np

x = np.array([[1,0,2],
            [1,1,2],
            [1,2,1],
            [1,-3,1],
            [1,-2,-1],
            [1,-3,-2]]).astype(float)
t = np.array([1,1,1,0,0,0]).astype(float)
w = np.array([1.5, -4.5, -2.5]).astype(float)
lr = 1.0

def H(y):
    if y > 0:
        return 1
    elif y == 0:
        return 0.5
    else:
        return 0

for i in range(2):
    for j in range(len(x)):
        y = H(np.matmul(w, x[j].T))
        w += lr * (t[j] - y) * x[j]
        print(y, w)