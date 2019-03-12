"""
1.  Pick any dataset from the dataset sheet in the class sheet or online which
ncludes both numeric and non-numeric features

     a. Perform exploratory data analysis on the data set (like Handling null values,
removing the features not correlated to the target class, encoding the categorical features, …)

     b. Apply the three classification algorithms Naïve Baye’s, SVM and KNN on the
chosen data set and report which classifier gives better result.

# The dataset I use for question 1 is car.txt

| class values
unacc, acc, good, vgood
| attributes
buying:   vhigh, high, med, low.
maint:    vhigh, high, med, low.
doors:    2, 3, 4, 5more.
persons:  2, 4, more.
lug_boot: small, med, big.
safety:   low, med, high.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv('car.txt', sep=",", header=None)
dataset.columns = ["buying", "maint", "doors", "persons", "lugBoot", "safety", "classValue"]
# dataset.info()
# print(dataset.isnull().sum())

# Transforming and engineering non-numeric features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Num_dataset = dataset
# Num_dataset['numBuying'] = le.fit_transform(dataset['buying'].astype('str'))

Num_dataset['numBuying'] = dataset['buying'].str.replace('low', '0')
Num_dataset['numBuying'] = Num_dataset['numBuying'].str.replace('vhigh', '3')
Num_dataset['numBuying'] = Num_dataset['numBuying'].str.replace('med', '1')
Num_dataset['numBuying'] = Num_dataset['numBuying'].str.replace('high', '2').astype(int)

# Num_dataset['numMaint'] = le.fit_transform(dataset['maint'].astype('str'))
# Num_dataset['numMaint'] = dataset['maint'].map({'low': 0, 'med': 1, 'high': 2, 'v-high': 3}).astype(int)
Num_dataset['numMaint'] = dataset['maint'].str.replace('low', '0')
Num_dataset['numMaint'] = Num_dataset['numMaint'].str.replace('med', '1')
Num_dataset['numMaint'] = Num_dataset['numMaint'].str.replace('vhigh', '3')
Num_dataset['numMaint'] = Num_dataset['numMaint'].str.replace('high', '2').astype(int)


Num_dataset['numDoors'] = dataset['doors'].str.replace('more', '').astype(int)
Num_dataset['numPersons'] = dataset['persons'].str.replace('more', '6').astype(int)

# Num_dataset['numLugBoot'] = le.fit_transform(dataset['lugBoot'].astype('str'))
# Num_dataset['numLugBoot'] = dataset['lugBoot'].map({'small': 0, 'med': 1, 'big': 2}).astype(int)
Num_dataset['numLugBoot'] = dataset['lugBoot'].str.replace('big', '2')
Num_dataset['numLugBoot'] = Num_dataset['numLugBoot'].str.replace('med', '1')
Num_dataset['numLugBoot'] = Num_dataset['numLugBoot'].str.replace('small', '0').astype(int)

# Num_dataset['numSafety'] = le.fit_transform(dataset['safety'].astype('str'))
# Num_dataset['numSafety'] = dataset['safety'].map({'low': 0, 'med': 1, 'high': 2}).astype(int)
Num_dataset['numSafety'] = dataset['safety'].str.replace('high', '2')
Num_dataset['numSafety'] = Num_dataset['numSafety'].str.replace('med', '1')
Num_dataset['numSafety'] = Num_dataset['numSafety'].str.replace('low', '0').astype(int)

# Num_dataset['numValue'] = le.fit_transform(dataset['classValue'].astype('str'))
# Num_dataset['numValue'] = dataset['classValue'].map({'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}).astype(int)
Num_dataset['numValue'] = dataset['classValue'].str.replace('unacc', '0')
Num_dataset['numValue'] = Num_dataset['numValue'].str.replace('vgood', '3')
Num_dataset['numValue'] = Num_dataset['numValue'].str.replace('acc', '1')
Num_dataset['numValue'] = Num_dataset['numValue'].str.replace('good', '2').astype(int)

buy_corr = Num_dataset[['buying', 'numValue']].groupby(['buying'], as_index=False).mean().sort_values(by='numValue', ascending=False)
maint_corr = Num_dataset[["maint", "numValue"]].groupby(['maint'], as_index=False).mean().sort_values(by='numValue', ascending=False)
door_corr = Num_dataset[["doors", "numValue"]].groupby(['doors'], as_index=False).mean().sort_values(by='numValue', ascending=False)
person_corr = Num_dataset[["persons", "numValue"]].groupby(['persons'], as_index=False).mean().sort_values(by='numValue', ascending=False)
boot_corr = Num_dataset[["lugBoot", "numValue"]].groupby(['lugBoot'], as_index=False).mean().sort_values(by='numValue', ascending=False)
safe_corr = Num_dataset[["safety", "numValue"]].groupby(['safety'], as_index=False).mean().sort_values(by='numValue', ascending=False)

# Correcting by dropping features
print("The correlation are \n ===BUY===\n", buy_corr, "\n===maint===\n", maint_corr, "\n===door===\n", door_corr,
      "\n===person===\n", person_corr, "\n===boot===\n", boot_corr, "\n===safe===\n", safe_corr)

g = sns.FacetGrid(Num_dataset, col='classValue')
g.map(plt.hist, 'numBuying')
plt.show()
g = sns.FacetGrid(Num_dataset, col='classValue')
g.map(plt.hist, 'numMaint')
plt.show()
g = sns.FacetGrid(Num_dataset, col='classValue')
g.map(plt.hist, 'numDoors')
plt.show()
g = sns.FacetGrid(Num_dataset, col='classValue')
g.map(plt.hist, 'numPersons')
plt.show()
g = sns.FacetGrid(Num_dataset, col='classValue')
g.map(plt.hist, 'numLugBoot')
plt.show()
g = sns.FacetGrid(Num_dataset, col='classValue')
g.map(plt.hist, 'numSafety')
plt.show()

# dataset.info()

data_train = Num_dataset.drop(["buying", "maint", "doors", "persons", "lugBoot", "safety", "classValue"], axis=1)
x = data_train.drop("numValue", axis=1)
y = dataset.classValue
# x.info()
# y.info()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.33)

# NB
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(X_train, y_train)
print("\n----------GNB----------")
# GaussianNB(priors=None, var_smoothing=1e-09)
print('Accuracy of Naive Bayes GaussianNB on training set: {:.2f}'.format(GNB.score(X_train, y_train)))
# Evaluate the model on testing part
print('Accuracy of Naive Bayes GaussianNB on test set: {:.2f}'.format(GNB.score(X_test, y_test)))

# SVM
from sklearn.svm import SVC
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
print("\n----------SVM----------")
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
# test data set acc
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
print("\n----------KNN----------")
print('Accuracy of KNN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
# test data set acc
print('Accuracy of KNN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))

