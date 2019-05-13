import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder
from keras import metrics
import pickle
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
from sklearn import tree
import time
from keras.models import model_from_yaml


all_data = pd.read_csv("./creditcard.csv")

# " === EDA === "
# print(all_data.head())
# print(all_data.describe())

# "1.1 Null Value Check ==="
# print(all_data.isnull().sum())

# "1.2 Numeric Value --- No need"

# "1.3 Class dataset --- unbalanced"
# print(all_data['Class'].value_counts(dropna='False'))

# "1.5 sns show data correlation"
# # g = sns.FacetGrid(all_data, col='Class')
# # g.map(plt.hist, 'Time', bins=20)
# # plt.show()

# " kaggle - time - Nothing"
# f, (time_1, time_2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
# time_1.scatter(all_data.Time[all_data.Class == 1], all_data.Amount[all_data.Class == 1])
# time_1.set_title('Fraud')
# time_2.scatter(all_data.Time[all_data.Class == 0], all_data.Amount[all_data.Class == 0])
# time_2.set_title('Normal')
# plt.ylabel('Amount')
# plt.xlabel('Time')
# plt.show()
#
# " deal with the PCA 28 features"
# pca_f = all_data.ix[:, 1:15].columns
# plt.figure(figsize=(16, 7*14))
# g_pca = gridspec.GridSpec(7, 2)
# for k, idx in enumerate(all_data[pca_f]):
#     ax = plt.subplot(g_pca[k])
#     sns.distplot(all_data[idx][all_data.Class == 1], bins=40)
#     sns.distplot(all_data[idx][all_data.Class == 0], bins=40)
#     ax.set_title('PCA Feature: ' + str(idx))
# plt.show()
#
# pca_f1 = all_data.ix[:, 15:29].columns
# plt.figure(figsize=(16, 7*14))
# g_pca = gridspec.GridSpec(7, 2)
# for k, idx in enumerate(all_data[pca_f1]):
#     ax = plt.subplot(g_pca[k])
#     sns.distplot(all_data[idx][all_data.Class == 1], bins=40)
#     sns.distplot(all_data[idx][all_data.Class == 0], bins=40)
#     ax.set_title('PCA Feature: ' + str(idx))
# plt.show()

# " Corr"
# corr = all_data.corr()
# print(corr['Class'].sort_values(ascending=False)[:30], '\n')
# print(corr['Class'].sort_values(ascending=False)[-14:])


# " drop the features with low correlation wi"

all_data = all_data.drop(['Time', 'Amount', 'V28', 'V27', 'V26', 'V25', 'V24', 'V23', 'V22', 'V20', 'V15', 'V13', 'V8'],
                         axis=1)
# Create dataframes of only Fraud and Normal transactions.
all_data['Class'] = all_data['Class'].map({1: 0, 0: 1})
Normal = all_data[all_data.Class == 1]
Fraud = all_data[all_data.Class == 0]

# Set training X equal to 70% of the fraudulent data.
X_train = Fraud.sample(frac=0.7)
Normal_sam = Normal.sample(frac=0.7)

new_bal_f = Fraud.loc[~Fraud.index.isin(X_train.index)]
num_new_bal = len(new_bal_f)
new_nor = Normal.loc[~Normal.index.isin(Normal_sam.index)]

New_bal = pd.concat([new_bal_f, new_nor.sample(n=num_new_bal, random_state=1)], axis=0)
New_bal = shuffle(New_bal)

# Add the other 70% data of normal into X_train
# Normal_sam = Normal_sam.sample(n=400, random_state=1)
X_train = pd.concat([X_train, Normal_sam], axis=0)
# print(len(X_train))
Balance_data = Normal.sample(n=492, random_state=1)
Balance_data = pd.concat([Balance_data, Fraud.sample(n=492, random_state=1)], axis=0)
Balance_data = shuffle(Balance_data)

# X_test will have all the data except X_train
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

# balanced dataset build
X_Balance_test = Balance_data.drop(['Class'], axis=1)
y_Balance_test = Balance_data.Class

# New balanced dataset
X_new_bal = New_bal.drop(['Class'], axis=1)
y_new_bal = New_bal.Class

# print(len(X_Balance_test))
# print(len(y_Balance_test))

# Check to ensure all of the training/testing dataframes are of the correct length
# print(len(X_train))
# print(len(y_train))
# print(len(X_test))
# print(len(y_test))
# print(X_test)
# print(y_test)

class_names = np.array(['Fraud', 'Normal'])

'Deep Learning'
start_time_DL = time.time()

epochs = 15
batch_size = 64
model = Sequential()  # create model
model.add(Dense(20, input_dim=17, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # output layer

model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])

hist = model.fit(X_train, y_train,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(X_new_bal, y_new_bal))

print("Time consuming of Keras is: ", time.time() - start_time_DL)

loss, ac = model.evaluate(X_new_bal, y_new_bal, verbose=0)
print("The loss is: ", loss)
print("The accuracy is: ", ac)

model_ori = open('model_DL.pckl', 'wb')
pickle.dump(hist.history, model_ori)
model_ori.close()
# serialize model to YAML
model_yaml = model.to_yaml()
with open("model_DL.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model_DL.h5")

# # acc
# plt.plot(hist.history['val_acc'])
# plt.title('model DL')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['val acc'], loc='upper left')
# plt.show()
#
# # loss
# plt.plot(hist.history['val_loss'])
# plt.title('model DL')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['val loss'], loc='upper left')
# plt.show()

# from sklearn.metrics import roc_curve
# y_pred_keras = model.predict(X_new_bal).ravel()
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_new_bal, y_pred_keras)
# from sklearn.metrics import auc
# auc_keras = auc(fpr_keras, tpr_keras)
#
# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.show()

# load YAML and create model
# yaml_file = open('model_DL.yaml', 'r')
# loaded_model_yaml = yaml_file.read()
# yaml_file.close()
# loaded_model = model_from_yaml(loaded_model_yaml)
# # load weights into new model
# loaded_model.load_weights("model_DL.h5")
# print("Loaded model from disk")
#
# from sklearn.metrics import roc_curve
# y_pred_keras = loaded_model.predict(X_new_bal).ravel()
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_new_bal, y_pred_keras)
# from sklearn.metrics import auc
# auc_keras = auc(fpr_keras, tpr_keras)
#
# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.show()
#
#
# def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if not title:
#         if normalize:
#             title = 'Normalized confusion matrix'
#         else:
#             title = 'Confusion matrix, without normalization'
#
#     # Compute confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
#     # We want to show all ticks...
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            # ... and label them with the respective list entries
#            xticklabels=classes, yticklabels=classes,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')
#
#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
#
#     # Loop over data dimensions and create text annotations.
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#     return ax
#
#
# 'Confusion Matrix'
# y_pred_nb = loaded_model.predict_classes(X_new_bal)
# # Plot non-normalized confusion matrix
# plot_confusion_matrix(y_new_bal, y_pred_nb, classes=class_names,
#                       title='NB Confusion matrix')
# plt.show()


