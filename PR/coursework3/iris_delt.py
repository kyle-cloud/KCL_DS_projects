import numpy as np
import scipy.io

mat = scipy.io.loadmat('C:\\Users\\11394\\Documents\\GitHub\\KCL_DS_projects\\PR\\coursework3\\iris_class1_2_3_4D.mat') 
# create y and target
x = np.ones(len(mat['X'].T))
x = np.insert(mat['X'].T, 0, values=x, axis=1)
x = x.astype(float)

t = np.squeeze(mat['t']).astype(float)
t[t != 0] = -1
t[t == 0] = 1
t[t == -1] = 0

w = np.array([0.5, 2.5, -0.5, -3.5, 2.5]).astype(float)
lr = 0.1

def H(y):
    if y > 0:
        return 1
    elif y == 0:
        return 0.5
    else:
        return 0

# predict before learning
cnt = 0
for i in range(len(x)):
    if H(np.matmul(w, x[i].T)) == t[i]:
        cnt += 1
print(cnt / len(x))


for i in range(2):
    for j in range(len(x)):
        y = H(np.matmul(w, x[j].T))
        w += lr * (t[j] - y) * x[j]
print(w)

# predict after learning
cnt = 0
for i in range(len(x)):
    if H(np.matmul(w, x[i].T)) == t[i]:
        cnt += 1
print(cnt / len(x))