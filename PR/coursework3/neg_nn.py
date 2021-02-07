import numpy as np

W = np.array([[1,1,0],
    [1,1,1]]).astype(float)
x = np.array([1,1,0]).astype(float)
alpha = 0.25
y = np.array([0,0]).astype(float)
e = x

for _ in range(5):
    We = np.matmul(W, e)
    y += alpha * We.T
    WTy = np.matmul(W.T, y)
    print(e, We, y, WTy)
    e =x - WTy
print(y)