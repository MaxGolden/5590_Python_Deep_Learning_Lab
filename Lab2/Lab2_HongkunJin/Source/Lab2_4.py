import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

diabetes = load_diabetes()
dataset = pd.DataFrame(data=np.c_[diabetes['data'], diabetes['target']],
                     columns=diabetes['feature_names'] + ['target'])

dataset.info()
x = dataset.drop(['target'], axis=1)
# x = dataset.drop(['target', 's1', 's2', 'age', 'sex'], axis=1)
y = np.log(dataset.target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.33)

corr = dataset.corr()
print(corr['target'].sort_values(ascending=False)[:6], '\n')
print(corr['target'].sort_values(ascending=False)[-6:])

quality_pivot = dataset.pivot_table(index='s1', values='target', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.show()
quality_pivot = dataset.pivot_table(index='age', values='target', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.show()
quality_pivot = dataset.pivot_table(index='s2', values='target', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.show()
quality_pivot = dataset.pivot_table(index='sex', values='target', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.show()

from sklearn import linear_model
lr1 = linear_model.LinearRegression()
model = lr1.fit(X_train, y_train)

print('r2 is: ', model.score(X_test, y_test))
prediction = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print('rmse: ', mean_squared_error(y_test, prediction))



