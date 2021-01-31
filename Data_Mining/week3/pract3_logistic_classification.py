import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

# load data
iris = load_iris()
M = len(iris.data)
print('classes = ', iris.target_names)
print('attributes = ', iris.feature_names)
print('number of instances = %d' % M)

X_train, X_test, y_train, y_test = train_test_split(iris.data[:, :2], iris.target, random_state=0)

# build tree
clf = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')

# fit the data
clf.fit(X_train, y_train)

# evaluate
y_hat = clf.predict(X_test)

print('accuracy = %f' % (sum(y_hat == y_test) / len(y_test) * 100))
print('train score = %f' % clf.score(X_train, y_train))
print('test score = %f' % clf.score(X_test, y_test))
print('accuracy score = %f' % metrics.accuracy_score(y_test, y_hat))
print('confusion matrix: \n', metrics.confusion_matrix(y_test, y_hat))
print('precision = ', metrics.precision_score(y_test, y_hat, average=None))
print('recall = ', metrics.recall_score(y_test, y_hat, average=None))
print('f1 = ', metrics.f1_score(y_test, y_hat, average=None))

# decision boundary
step_size = 0.01
x0_min = min(iris.data[:, 0])
x0_max = max(iris.data[:, 0])
x1_min = min(iris.data[:, 1])
x1_max = max(iris.data[:, 1])

X_pairs = np.array([[i, j] for j in np.arange(x1_min - step_size, x1_max + step_size, step_size) 
                            for i in np.arange(x0_min - step_size, x0_max + step_size, step_size)]).astype(float)
y_hat_pairs = clf.predict(X_pairs)
print('mesh score = ', clf.score(X_pairs, y_hat_pairs))

x0_range = np.arange(x0_min - step_size, x0_max + step_size, step_size) 
x1_range = np.arange(x1_min - step_size, x1_max + step_size, step_size)
x0_mesh, x1_mesh = np.meshgrid(x0_range, x1_range)
y_hat_mesh = y_hat_pairs.reshape(x0_mesh.shape)

plt.set_cmap('Blues')
plt.pcolormesh(x0_mesh, x1_mesh, y_hat_mesh, shading='flat')
plt.scatter(iris.data[iris.target == 0][:, 0], iris.data[iris.target == 0][:, 1], c='red', marker='.')
plt.scatter(iris.data[iris.target == 1][:, 0], iris.data[iris.target == 1][:, 1], c='green', marker='>')
plt.scatter(iris.data[iris.target == 2][:, 0], iris.data[iris.target == 2][:, 1], c='blue', marker='s')
plt.show()

# evaluate with roc
import sklearn.preprocessing as preprocess

conf_scores = clf.decision_function(X_pairs)
y_binary = preprocess.label_binarize(y_hat_pairs, classes=sorted(set(iris.target)))

fpr = dict()
tpr = dict()
for c in range(3):
    fpr[c], tpr[c], tmp = metrics.roc_curve(y_binary[:, c], conf_scores[:, c])
for c in range(3):
    plt.plot(fpr[c], tpr[c], label=iris.target_names[c])
plt.xlabel('false positive(FP) rate')
plt.ylabel('true positive(TP) rate')
plt.legend()
plt.show()