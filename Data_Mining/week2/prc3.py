####################################
#3.1 load ans plot data 
####################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('F:\OneDrive\Aclass\S2_Data_mining\week2\data\london-borough-profiles-jan2018.csv', encoding='ISO-8859-1')
data.head()
data = data.replace('.', 'NaN')
x, y = data.iloc[:, 70].astype(float), data.iloc[:, 71].astype(float)

plt.figure()
plt.scatter(x, y)
plt.margins(0.5)
plt.title('raw data')
plt.xlabel('age(men)')
plt.ylabel('age(women)')
plt.show()



####################################
#3.2 patition data and plot
####################################
from sklearn import model_selection

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)

fig, ax = plt.subplots()
ax.scatter(x_train, y_train, c='blue', label='train')
ax.scatter(x_test, y_test, c='red', marker='s', label='test')
ax.margins(0.5)
ax.set_title('partitioned data')
ax.legend()
ax.set_xlabel('age(men)')
ax.set_ylabel('age(women)')
fig.show()



####################################
#3.3 generate some random data for regression
####################################
from sklearn import datasets

x, y, p = datasets.make_regression(n_samples=100, n_features=1, n_informative=1, noise=10, coef=True)
fig, ax = plt.subplots()
ax.scatter(x, y, c='blue')
ax.margins(0.1)
ax.set_title('raw data')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.show()

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)
fig, ax = plt.subplots()
ax.scatter(x_train, y_train, c='blue', label='train')
ax.scatter(x_test, y_test, c='red', marker='s', label='test')
ax.margins(0.1)
ax.set_title('partitioned data')
ax.legend(loc='upper left') 
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.show()