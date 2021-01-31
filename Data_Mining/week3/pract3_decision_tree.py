import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics

# load data
iris = load_iris()
M = len(iris.data)
print('classes = ', iris.target_names)
print('attributes = ', iris.feature_names)
print('number of instances = %d' % M)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

# build tree
clf = tree.DecisionTreeClassifier(random_state=0)

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

# visualization
## raw decision tree path
print(clf.decision_path(iris.data))
## GraphViz
