import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import datasets


####################################
# Load Data
####################################
data = pd.read_csv('F:\OneDrive\Aclass\S2_Data_mining\week2\data\london-borough-profiles-jan2018.csv', encoding='ISO-8859-1')
data.head()
data = data.replace('.', 'NaN')
x, y = data.iloc[:, 70].astype(float), data.iloc[:, 71].astype(float)
x = x.fillna(x.mean()).values.reshape(-1, 1)
y = y.fillna(y.mean()).values.reshape(-1, 1)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)


####################################
# change data
####################################
x_self, y_self, p = datasets.make_regression(n_samples=100, n_features=1, n_informative=1, noise=10, coef=True)
y_self = y_self.reshape(-1, 1)
x_self_train, x_self_test, y_self_train, y_self_test = model_selection.train_test_split(x_self, y_self, test_size=0.1)


####################################
# create model
####################################
def gradient_descent_2(M, X, w, y, alpha):
    for j in range(M):
        y_hat = w[0] + w[1] * X[j]
        theta = y[j] - y_hat
        w[0] = w[0] + alpha * theta / M
        w[1] = w[1] + alpha * theta * X[j] / M
    return w

def compute_error(M, X, w, y):
    error = 0
    for j in range(M):
        y_hat = w[0] + w[1] * X[j]
        error += (y[j] - y_hat) ** 2
    error /= M
    return error

def compute_r2(M, X, w, y):
    u = 0
    for j in range(M):
        y_hat = w[0] + w[1] * X[j]
        u += (y[j] - y_hat) ** 2
    y_mean = sum(y) / M
    v = sum((y - y_mean) ** 2)
    R2 = 1 - u / v
    return R2

def plot_line(X, y, y_hat, epoch, error, R2):
    fig, ax = plt.subplots()
    ax.scatter(X, y, c='blue')
    ax.plot(X, y_hat, '-')
    ax.margins(0.2)
    ax.set_title('after %d iteration\terror = %.3lf\tR^2 = %.3lf' % (epoch, error, R2))
    fig.show()

def plot_R2(R2):
    fig, ax = plt.subplots()
    ax.plot(np.array(range(len(R2_list))), R2, '-')
    fig.show()


####################################
# train model
####################################
epoches = 1000
alpha = 0.01
R2_list = []
w = np.random.randn(2, 1)

def train(X, y, w):
    for epoch in range(epoches):
        w = gradient_descent_2(len(X), X, w, y, alpha)
        error = compute_error(len(X), X, w, y)
        R2 = compute_r2(len(X), X, w, y)
        R2_list.append(R2)
        if epoch in [0, 1, 2, 3]: plot_line(X, y, X * w[1] + w[0], epoch, error, R2)
        if error < 200: print('converged at %d' % epoch); break
    error = compute_error(len(X), X, w, y)
    R2 = compute_r2(len(X), X, w, y)
    plot_line(X, y, X * w[1] + w[0], epoches, error, R2)
    plot_R2(R2_list)


####################################
# test model
####################################
def test(X, y, w):
    error = compute_error(len(X), X, w, y)
    R2 = compute_r2(len(X), X, w, y)
    plot_line(X, y, X * w[1] + w[0], epoches, error, R2)


####################################
# __main__
####################################
# train(x_train, y_train, w)
# test(x_test, y_test, w)

train(x_self_train, y_self_train, w)
test(x_self_test, y_self_test, w)


####################################
# test all the  hyper-parameter
####################################
# alpha - w - p




####################################
# scikit-learn
####################################
from sklearn import linear_model
from sklearn import metrics

lr = linear_model.LinearRegression()
lr.fit(x_self_train, y_self_train)
print('sklearn regression equation: y = %f + %fx' % (lr.intercept_, lr.coef_[0]))

y_hat = lr.predict(x_self_train)
R2 = metrics.r2_score(y_self_train, y_hat)
error = metrics.mean_squared_error(y_self_train, y_hat)
print('R2 = %f' % R2)
print('mean squared error = ' % error)
plot_line(x_self_train, y_self_train, y_hat, 0, error, R2)

y_hat = lr.predict(x_self_test)
R2 = metrics.r2_score(y_self_test, y_hat)
error = metrics.mean_squared_error(y_self_test, y_hat)
print('R2 = %f' % R2)
print('mean squared error = ' % error)
plot_line(x_self_test, y_self_test, y_hat, 0, error, R2)