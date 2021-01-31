import numpy as np
from collections import defaultdict
from sklearn.datasets import load_iris

# load data
iris = load_iris()
M = len(iris.data)
print('classes = ', iris.target_names)
print('attributes = ', iris.feature_names)
print('number of instances = %d' % M)

X = iris.data
y = iris.target

# sepal length
dic_len = {}
for i in range(M):
    if X[i][0] not in dic_len:
        dic_len[X[i][0]] = [0,0,0]
    dic_len[X[i][0]][y[i]] += 1
dic_len_most = defaultdict(list)
for key, value in dic_len.items():
    dic_len_most[key] = np.argmax(dic_len[key])

# sepal width
dic_wid = {}
for i in range(M):
    if X[i][1] not in dic_wid:
        dic_wid[X[i][1]] = [0,0,0]
    dic_wid[X[i][1]][y[i]] += 1
dic_wid_most = defaultdict(list)
for key, value in dic_wid.items():
    dic_wid_most[key] = np.argmax(dic_wid[key])

# all the combination
dic_comb = defaultdict(list)
for key_len in dic_len.keys():
    for key_wid in dic_wid.keys():
        label_freq_len = [dic_len_most[key_len], np.max(dic_len[key_len])]
        label_freq_wid = [dic_wid_most[key_wid], np.max(dic_wid[key_wid])]
        dic_comb[(key_len, key_wid)] = label_freq_len[0] if label_freq_len[1] >= label_freq_wid[1] else label_freq_wid[0]

# evaluation
correct = 0
for i in range(M):
    if dic_comb[(X[i][0], X[i][1])] == y[i]:
        correct += 1
print('accuracy = %.2f' % (correct / M * 100))