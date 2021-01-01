import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import datasets


####################################
# Load Data
####################################
data = pd.read_csv('F:\OneDrive\Aclass\S2_Data_mining\week2\prac2-data\london-borough-profiles-jan2018.csv', encoding='ISO-8859-1')
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
    error = sum((y - X * w[1] + w[0]) ** 2)
    error = error / M
    return error

def compute_r2(M, X, w, y):
    u = sum((y - X * w[1] + w[0]) ** 2)
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

def plot_R2(R2, epoches):
    fig, ax = plt.subplots()
    ax.plot(np.array(range(epoches)), R2, '-')
    fig.show()


####################################
# train model
####################################
epoches = 10000
alpha = 0.001
R2_list = []
w = np.random.randn(2, 1)

def train(X, y, w):
    for epoch in range(epoches):
        error = compute_error(len(X), X, w, y)
        R2 = compute_r2(len(X), X, w, y)
        R2_list.append(R2)
        if epoch in [1000, 2000, 4000, 8000]: plot_line(X, y, X * w[1] + w[0], epoch, error, R2)
        w = gradient_descent_2(len(X), X, w, y, alpha)
    error = compute_error(len(X), X, w, y)
    R2 = compute_r2(len(X), X, w, y)
    plot_line(X, y, X * w[1] + w[0], epoches, error, R2)
    plot_R2(R2_list, epoches)


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