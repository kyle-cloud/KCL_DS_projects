# classification
## load 'adult.csv' and drop 'fnlwgt' 
import numpy as np
import pandas as pd

data = pd.read_csv('F:\OneDrive\Aclass\S2_Data_mining\courseworks\coursework1\\adult.csv', encoding='ISO-8859-1')
data.drop(['fnlwgt'], axis=1, inplace=True)


## 1. create an overview table
num_instances = len(data)
num_missing = data.isna().sum() #.sum()
frac_missing = num_missing / num_instances
num_ins_missing = data.isna().T.any().sum()
frac_ins_missing = num_ins_missing / num_instances


## 2. LabelEncoder
from sklearn.preprocessing import LabelEncoder

X = data.drop(['class'], axis=1)
le = LabelEncoder()
X = X.apply(lambda col: le.fit_transform(col.astype(str)))
discrete_values = X.apply(lambda col: set(col))
print(discrete_values)


## 3. decision tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split

data_cp = data[data.isna().T.any() == False]
y = data_cp['class']
X = data_cp.drop(['class'], axis=1)
X = X.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y)
# clf.score(X_test, y_test)
y_hat = clf.predict(X)
# print(cross_val_score(clf, X, y, cv=10))
print("error rate is %f" % (np.sum([y_hat != y]) / len(y)))


## 4. process missing values
missing_df = data[data.isna().T.any()]
non_missing_df = data[data.isna().T.any() == False].sample(len(missing_df), random_state=0)
D_ = pd.concat([missing_df, non_missing_df], axis=0)
D_ = D_.sample(frac=1).reset_index(drop=True)

D_1 = D_.fillna('missing')
D_2 = D_.apply(lambda col: col.fillna(col.value_counts().idxmax()))

y_1 = D_1['class'] # tree for D_1
X_1 = D_1.drop(['class'], axis=1)
X_1 = X_1.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)))
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_1, y_1)
y_hat = clf.predict(X)
print("error rate is %f" % (np.sum([y_hat != y]) / len(y)))

y_2 = D_2['class'] # tree for D_2
X_2 = D_2.drop(['class'], axis=1)
X_2 = X_2.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)))
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_2, y_2)
y_hat = clf.predict(X)
print("error rate is %f" % (np.sum([y_hat != y]) / len(y)))
