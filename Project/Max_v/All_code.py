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


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def roc_curve_plot(test_x, test_y, model_s, model1, model2, model3, model4, model5, model6, DL_model):

    y_scores_s = model_s.decision_function(test_x)
    y_scores1 = model1.predict_proba(test_x)
    y_scores2 = model2.predict_proba(test_x)
    y_scores3 = model3.predict_proba(test_x)
    y_scores4 = model4.predict_proba(test_x)
    y_scores5 = model5.predict_proba(test_x)
    y_scores6 = model6.predict_proba(test_x)
    y_scores_dl = DL_model.predict(test_x).ravel()

    fpr_s, tpr_s, thresholds_s = roc_curve(test_y, y_scores_s, pos_label=1)
    fpr1, tpr1, thresholds1 = roc_curve(test_y, y_scores1[:, 1], pos_label=1)
    fpr2, tpr2, thresholds2 = roc_curve(test_y, y_scores2[:, 1], pos_label=1)
    fpr3, tpr3, thresholds3 = roc_curve(test_y, y_scores3[:, 1], pos_label=1)
    fpr4, tpr4, thresholds4 = roc_curve(test_y, y_scores4[:, 1], pos_label=1)
    fpr5, tpr5, thresholds5 = roc_curve(test_y, y_scores5[:, 1], pos_label=1)
    fpr6, tpr6, thresholds6 = roc_curve(test_y, y_scores6[:, 1], pos_label=1)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_new_bal, y_scores_dl)

    roc_auc_s = auc(fpr_s, tpr_s)
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)
    roc_auc3 = auc(fpr3, tpr3)
    roc_auc4 = auc(fpr4, tpr4)
    roc_auc5 = auc(fpr5, tpr5)
    roc_auc6 = auc(fpr6, tpr6)
    roc_auc_dl = auc(fpr_keras, tpr_keras)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_s, tpr_s, 'b', label='SVM (area = {:.3f})'.format(roc_auc_s))
    plt.plot(fpr1, tpr1, 'r', label='NB (area = {:.3f})'.format(roc_auc1))
    plt.plot(fpr2, tpr2, 'y', label='KNN (area = {:.3f})'.format(roc_auc2))
    plt.plot(fpr3, tpr3, 'pink', label='DT (area = {:.3f})'.format(roc_auc3))
    plt.plot(fpr4, tpr4, 'cyan', label='RF (area = {:.3f})'.format(roc_auc4))
    plt.plot(fpr5, tpr5, 'purple', label='ET (area = {:.3f})'.format(roc_auc5))
    plt.plot(fpr6, tpr6, 'g', label='LR (area = {:.3f})'.format(roc_auc6))
    plt.plot(fpr_keras, tpr_keras, 'gold', label='Keras (area = {:.3f})'.format(roc_auc_dl))

    # plt.legend(["SVM AUC:%0.2f" % roc_auc_s, "NB", "KNN", "DT", "RF", "ET", "LR"], loc='lower right')
    plt.legend(loc='best')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve of Models')
    plt.show()


# def precision_recall_curve(test_x, test_y, model):
#     y_scores = model.predict_proba(test_x)
#     from sklearn.metrics import average_precision_score
#     average_precision = average_precision_score(test_y, y_scores[:, 1])
#     print('Average precision-recall score: {0:0.2f}'.format(
#           average_precision))
#     from sklearn.metrics import precision_recall_curve
#     from sklearn.utils.fixes import signature
#     precision, recall, _ = precision_recall_curve(test_y, y_scores[:, 1])
#     # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
#     step_kwargs = ({'step': 'post'}
#                    if 'step' in signature(plt.fill_between).parameters
#                    else {})
#     plt.step(recall, precision, color='b', alpha=0.2,
#              where='post')
#     plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
#
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#     plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
#               average_precision))
#     plt.show()


def roc_curve_plot_svm(test_x, test_y, model):
    y_scores = model.decision_function(test_x)
    fpr, tpr, thresholds = roc_curve(test_y, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve of GNB')
    plt.show()


# def precision_recall_curve_svm(test_x, test_y, model):
#     y_scores = model.decision_function(test_x)
#     from sklearn.metrics import average_precision_score
#     average_precision = average_precision_score(test_y, y_scores)
#     print('Average precision-recall score: {0:0.2f}'.format(
#           average_precision))
#     from sklearn.metrics import precision_recall_curve
#     from sklearn.utils.fixes import signature
#     precision, recall, _ = precision_recall_curve(test_y, y_scores)
#     # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
#     step_kwargs = ({'step': 'post'}
#                    if 'step' in signature(plt.fill_between).parameters
#                    else {})
#     plt.step(recall, precision, color='b', alpha=0.2,
#              where='post')
#     plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
#
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#     plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
#               average_precision))
#     plt.show()


"2.1 NB"
start_time = time.time()
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
model_GNB = GNB.fit(X_train, y_train)
print("\n----------GNB----------")
print('Accuracy of Naive Bayes GaussianNB on training set: {:.3f}'.format(GNB.score(X_train, y_train)))
# Evaluate the model on testing part
print('Accuracy of Naive Bayes GaussianNB on test set: {:.3f}'.format(GNB.score(X_new_bal, y_new_bal)))
' test data matrix'

# y_pred = model_GNB.predict(X_test)
# # Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names,
#                       title='Confusion matrix, without normalization')
# plt.show()

'Confusion Matrix'
y_pred_nb = GNB.predict(X_new_bal)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_new_bal, y_pred_nb, classes=class_names,
                      title='NB Confusion matrix')
print("Time consuming of NB is: ", time.time() - start_time)
plt.show()

# 'Roc for test'
# roc_curve_plot(X_new_bal, y_new_bal, GNB)
# 'Pre_re for test'
# precision_recall_curve(X_new_bal, y_new_bal, GNB)


"2.2 SVM"
start_time_svm = time.time()
from sklearn.svm import SVC
svm = SVC(kernel='rbf', gamma='scale')
svm.fit(X_train, y_train)
print("\n----------SVM----------")
print('Accuracy of SVM classifier on training set: {:.3f}'.format(svm.score(X_train, y_train)))
# test data set acc
print('Accuracy of SVM classifier on test set: {:.3f}'.format(svm.score(X_new_bal, y_new_bal)))

# ' test data matrix'
# y_pred = svm.fit(X_train, y_train).predict(X_test)
# # Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names,
#                       title='Confusion matrix, without normalization')
# plt.show()

'Confusion Matrix'
y_pred_svm = svm.predict(X_new_bal)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_new_bal, y_pred_svm, classes=class_names,
                      title='SVM Confusion matrix')
print("Time consuming of SVM is: ", time.time() - start_time_svm)
plt.show()

# '''''''important'
# y_score = svm.decision_function(X_test)
# ' roc for test'
# roc_curve_plot_svm(X_Balance_test, y_Balance_test, svm)
# ' pre_re for test'
# precision_recall_curve_svm(X_Balance_test, y_Balance_test, svm)

"2.3 KNN"
start_time_knn = time.time()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
print("\n----------KNN----------")
print('Accuracy of KNN classifier on training set: {:.3f}'.format(knn.score(X_train, y_train)))
# test data set acc
print('Accuracy of KNN classifier on test set: {:.3f}'.format(knn.score(X_new_bal, y_new_bal)))

# ' test data matrix'
# y_pred = knn.fit(X_train, y_train).predict(X_test)
# # Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names,
#                       title='Confusion matrix, without normalization')
# plt.show()

'Confusion Matrix'
y_pred_knn = knn.predict(X_new_bal)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_new_bal, y_pred_knn, classes=class_names,
                      title='KNN Confusion matrix')
print("Time consuming of KNN is: ", time.time() - start_time_knn)
plt.show()

# ' roc for test'
# roc_curve_plot(X_test, y_test, knn)
# ' pre_re for test'
# precision_recall_curve(X_Balance_test, y_Balance_test, knn)

"2.4 Decision Tree"
start_time_DT = time.time()
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

print("\n----------DT----------")
print('Accuracy of DT classifier on training set: {:.3f}'.format(clf.score(X_train, y_train)))
# test data set acc
print('Accuracy of DT classifier on test set: {:.3f}'.format(clf.score(X_new_bal, y_new_bal)))

# ' test data matrix'
# y_pred = clf.fit(X_train, y_train).predict(X_test)
# # Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names,
#                       title='Confusion matrix, without normalization')
# plt.show()

'Confusion Matrix'
y_pred_dt = clf.predict(X_new_bal)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_new_bal, y_pred_dt, classes=class_names,
                      title='DT Confusion matrix')
print("Time consuming of DT is: ", time.time() - start_time_DT)
plt.show()

# ' roc for test'
# roc_curve_plot(X_test, y_test, clf)
# ' pre_re for test'
# precision_recall_curve(X_Balance_test, y_Balance_test, clf)

"2.5 Random Forest"
start_time_RF = time.time()
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_train)

print("\n----------RF----------")
print('Accuracy of RF classifier on training set: {:.3f}'.format(rfc.score(X_train, y_train)))
# test data set acc
print('Accuracy of RF classifier on test set: {:.3f}'.format(rfc.score(X_new_bal, y_new_bal)))

# ' test data matrix'
# y_pred = rfc.fit(X_train, y_train).predict(X_test)
# # Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names,
#                       title='Confusion matrix, without normalization')
# plt.show()

'Confusion Matrix'
y_pred_rf = rfc.predict(X_new_bal)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_new_bal, y_pred_rf, classes=class_names,
                      title='RF Confusion matrix')
print("Time consuming of DT is: ", time.time() - start_time_RF)
plt.show()

# ' roc for test'
# roc_curve_plot(X_test, y_test, rfc)
# ' pre_re for test'
# precision_recall_curve(X_Balance_test, y_Balance_test, rfc)

"2.6 Extra Trees"
start_time_ET = time.time()
from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(n_estimators=10)
etc.fit(X_train, y_train)

print("\n----------ET----------")
print('Accuracy of ET classifier on training set: {:.3f}'.format(etc.score(X_train, y_train)))
# test data set acc
print('Accuracy of ET classifier on test set: {:.3f}'.format(etc.score(X_new_bal, y_new_bal)))

# ' test data matrix'
# y_pred = etc.fit(X_train, y_train).predict(X_test)
# # Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names,
#                       title='Confusion matrix, without normalization')
# plt.show()

'Confusion Matrix'
y_pred_et = etc.predict(X_new_bal)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_new_bal, y_pred_et, classes=class_names,
                      title='Confusion matrix, without normalization')
print("Time consuming of DT is: ", time.time() - start_time_ET)
plt.show()

# ' roc for test'
# roc_curve_plot(X_test, y_test, etc)
# ' pre_re for test'
# precision_recall_curve(X_Balance_test, y_Balance_test, etc)

"2.7 Logistic Regression"
start_time_LR = time.time()
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(solver='lbfgs')
LR.fit(X_train, y_train)

print("\n----------LR----------")
print('Accuracy of LR classifier on training set: {:.3f}'.format(LR.score(X_train, y_train)))
# test data set acc
print('Accuracy of LR classifier on test set: {:.3f}'.format(LR.score(X_new_bal, y_new_bal)))

# # ' test data matrix'
# y_pred = LR.fit(X_train, y_train).predict(X_test)
# # Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names,
#                       title='Confusion matrix, without normalization')
# plt.show()

'Confusion Matrix'
y_pred_LR = LR.predict(X_new_bal)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_new_bal, y_pred_LR, classes=class_names,
                      title='Confusion matrix, without normalization')
print("Time consuming of DT is: ", time.time() - start_time_LR)
plt.show()

# ' roc for test'
# roc_curve_plot(X_test, y_test, LR)
# ' pre_re for test'
# precision_recall_curve(X_Balance_test, y_Balance_test, LR)


"3.1 DL - Keras sequential"

yaml_file = open('model_DL.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
DL_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
DL_model.load_weights("model_DL.h5")
print("Loaded model from disk")
DL_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])

loss1, ac1 = DL_model.evaluate(X_train, y_train, verbose=0)
loss, ac = DL_model.evaluate(X_new_bal, y_new_bal, verbose=0)

print("\n----------Keras----------")
print('Accuracy of Keras classifier on training set: {:.3f}'.format(ac1))
# test data set acc
print('Accuracy of Keras classifier on test set: {:.3f}'.format(ac))

'Confusion Matrix'
y_pred_nb = DL_model.predict_classes(X_new_bal)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_new_bal, y_pred_nb, classes=class_names,
                      title='Keras Confusion matrix')
plt.show()


"ROC curve"
roc_curve_plot(X_new_bal, y_new_bal, svm, GNB, knn, clf, rfc, etc, LR, DL_model)


"""
False Positives & False Negatives
Confusion Matrix
Accuracy Paradox
Precision
Recall
F1 score

Nearest Neighbor (k-nearest neighbors (kNN))
Naive Bayes Classifier
Support Vector Machines (SVM)
Decision Trees
Boosted Trees
Random Forest
Neural Networks
"""


