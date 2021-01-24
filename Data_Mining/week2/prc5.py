from sklearn import datasets
from sklearn import model_selection
import matplotlib.pyplot as plt

# data
x, y = datasets.make_classification(n_features=1, n_redundant=0, n_informative=1, n_classes=2,
                                    n_clusters_per_class=1, n_samples=100, random_state=12)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)

fig, ax = plt.subplots()
ax.scatter(x_train[y_train == 0], x_train[y_train == 0], c='blue', marker='.', label='train, class0')
ax.scatter(x_train[y_train == 1], x_train[y_train == 1], c='red', marker='+', label='train, class1')
ax.scatter(x_test[y_test == 0], x_test[y_test == 0], c='blue', marker='>', label='test, class0')
ax.scatter(x_test[y_test == 1], x_test[y_test == 1], c='red', marker='s', label='test, class1')
ax.margins(0.5)
ax.legend()
fig.show()

# model
from sklearn import linear_model

per = linear_model.Perceptron()
per.fit(x_train, y_train)

# evaluation
from sklearn import metrics

y_hat = per.predict(x_train)
print('train accuracy: %f' % metrics.accuracy_score(y_train, y_train, normalize=True))
y_hat = per.predict(x_test)
print('test accuracy: %f' % metrics.accuracy_score(y_hat, y_test, normalize=True))



# trainset
y_plot = per.intercept_ + x_train * per.coef_[0, 0]
fig, ax = plt.subplots()
ax.scatter(x_train[y_train == 0], x_train[y_train == 0], c='blue', marker='.', label='train, class0')
ax.scatter(x_train[y_train == 1], x_train[y_train == 1], c='red', marker='+', label='train, class1')
ax.plot(x_train, y_plot, '-', c='black')
ax.margins(0.5)
ax.legend()
fig.show()

# testset
y_plot = per.intercept_ + x_test * per.coef_[0, 0]
fig, ax = plt.subplots()
ax.scatter(x_test[y_test == 0], x_test[y_test == 0], c='blue', marker='>', label='test, class0')
ax.scatter(x_test[y_test == 1], x_test[y_test == 1], c='red', marker='s', label='test, class1')
ax.plot(x_test, y_plot, '-', c='black')
ax.margins(0.5)
ax.legend()
fig.show()
