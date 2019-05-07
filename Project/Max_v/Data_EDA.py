import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_score

all_data = pd.read_csv("./creditcard.csv")
class_names = np.array(['Normal', 'Fraud'])

" === EDA === "
# print(all_data.head())
# print(all_data.describe())
"1.1 Null Value Check ==="
# print(all_data.isnull().sum())

"1.2 Numeric Value --- No need"
"1.3 Class dataset --- unbalanced"
# print(all_data['Class'].value_counts(dropna='False'))

"1.4 Info"
# all_data.info()

"1.5 sns show data correlation"
# g = sns.FacetGrid(all_data, col='Class')
# g.map(plt.hist, 'Time', bins=20)
# plt.show()

" deal with the PCA 28 features"
# pca_f = all_data.ix[:, 1:29].columns
" plot correlation between pca28 with class 7 7 7 7"
# plt.figure(figsize=(12, 28*6))
# g_pca = gridspec.GridSpec(28, 1)
# for k, idx in enumerate(all_data[pca_f]):
#     ax = plt.subplot(g_pca[k])
#     sns.distplot(all_data[idx][all_data.Class == 1], bins=40)
#     sns.distplot(all_data[idx][all_data.Class == 0], bins=40)
#     ax.set_title('PCA Feature: ' + str(idx))
# plt.show()

" drop the features with low correlation wi"
all_data = all_data.drop(['Time', 'Amount', 'V28', 'V27', 'V26', 'V25', 'V24', 'V23', 'V22', 'V20', 'V15', 'V13', 'V8'],
                         axis=1)

# Create dataframes of only Fraud and Normal transactions.
Fraud = all_data[all_data.Class == 1]
Normal = all_data[all_data.Class == 0]
# print(len(Fraud))
# print(len(Normal))

# Set X_train equal to 80% of the fraudulent transactions.
X_train = Fraud.sample(frac=0.7)

# Add 80% of the normal transactions to X_train.
X_train = pd.concat([X_train, Normal.sample(frac=0.7)], axis=0)

Balance_data = Normal.sample(n=492, random_state=1)
Balance_data = pd.concat([Balance_data, Fraud.sample(n=492, random_state=1)], axis=0)
Balance_data = shuffle(Balance_data)

# X_test contains all the transaction not in X_train.
X_test = all_data.loc[~all_data.index.isin(X_train.index)]

# Shuffle the data frames so that the training is done in a random order.
X_train = shuffle(X_train)
X_test = shuffle(X_test)

# Add our target features to y_train and y_test.
y_train = X_train.Class
y_test = X_test.Class

# Drop target features from X_train and X_test.
X_train = X_train.drop(['Class'], axis=1)
X_test = X_test.drop(['Class'], axis=1)

# New balanced dataset build
X_Balance_test = Balance_data.drop(['Class'], axis=1)
y_Balance_test = Balance_data.Class
print(len(X_Balance_test))
print(len(y_Balance_test))

# Check to ensure all of the training/testing dataframes are of the correct length
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

