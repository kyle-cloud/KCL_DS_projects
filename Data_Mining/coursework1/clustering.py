# classification
## load 'adult.csv' and drop 'fnlwgt' 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('F:\OneDrive\Aclass\S2_Data_mining\courseworks\coursework1\\wholesale_customers.csv', encoding='ISO-8859-1')
data.drop(['Channel', 'Region'], axis=1, inplace=True)

## 1. mean, min and max
means = data.mean()
mins = data.min()
maxs = data.max()

## 2. k-means
from sklearn.cluster import KMeans

clt = KMeans(n_clusters=3, random_state=0)
clt.fit(data)
prediction = clt.predict(data)

fig, axes = plt.subplots(5, 5, figsize=(20, 20))
fig.subplots_adjust(wspace=0.5, hspace=0.5)
for i in range(5):
    cnt = 0
    for j in range(i + 1, 6):
        axes[i, cnt].scatter(data.iloc[:, i], data.iloc[:, j], c=prediction)
        axes[i, cnt].set_xlabel(data.columns.values[i])
        axes[i, cnt].set_ylabel(data.columns.values[j], rotation=90)
        cnt += 1
    for j in range(cnt, 5):
        axes[i, j].axis('off')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig.show()