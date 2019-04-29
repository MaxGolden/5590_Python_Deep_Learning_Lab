"""
1. Build a Sequential model using keras to implement Linear Regression
with any data set of your choice except the datasets being discussed
in the class or used before

a. Show the graph on TensorBoard
b. Plot the loss and then change the below parameter and report your
view how the result changes in each case

a.	learning rate
b.	batch size
c.	optimizer
d.	activation function

"""
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

# load dataset
dataset = pd.read_csv("diamonds_q1.csv")
# print(dataset)
# dataset.info()  # No null value

# Wrangling the non-numeric Features
# cut           53940 non-null object
# color         53940 non-null object
# clarity       53940 non-null object

le = LabelEncoder()
dataset['enc_cut'] = le.fit_transform(dataset['cut'].astype('str'))
dataset['enc_color'] = le.fit_transform(dataset['color'].astype('str'))
dataset['enc_clarity'] = le.fit_transform(dataset['clarity'].astype('str'))

all_data = dataset.drop(['Unnamed: 0', 'price', 'cut', 'color', 'clarity'], axis=1)
# all_train_data.info()
all_price = dataset['price']

X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_price,
                                                    test_size=0.30, random_state=87)
np.random.seed(123)

# model 1
epochs = 20
learning_rate = 0.005
batch_size = 64
optimizer = 'Adamax'
activation_function = 'linear'

# model 2
# epochs = 20
# learning_rate = 0.001
# batch_size = 32
# optimizer = 'adam'
# activation_function = 'relu'

model = Sequential()  # create model
model.add(Dense(50, input_dim=9, activation=activation_function))  # hidden layer
model.add(Dropout(0.1))
model.add(Dense(20, activation=activation_function))
model.add(Dense(10, activation=activation_function))
# my_first_nn.add(Dense(10, activation='sigmoid')) # output layer
model.add(Dense(1, input_dim=9, activation="relu"))  # output layer
model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=[metrics.mae])

# J:\5590Dl\DL\Lab3\max_version\q1_1_logs
# cd /path/to/log
# tensorboard --logdir=./
tensorboard = TensorBoard(log_dir='./q1_3_logs', histogram_freq=0,
                          write_graph=True, write_images=False)

hist = model.fit(X_train, Y_train,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(X_test, Y_test),
                 callbacks=[tensorboard])

mae, loss = model.evaluate(X_test, Y_test, verbose=0)
print("The mae is: ", mae)
print("The loss is: ", loss)

# # loss history
# plt.plot(hist.history['mean_absolute_error'])
# plt.plot(hist.history['val_mean_absolute_error'])
# plt.title('model mae')
# plt.ylabel('mae')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # acc
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# save
# serialize model to YAML

# model_yaml = model.to_yaml()
# with open("model_q1_2.yaml", "w") as yaml_file:
#     yaml_file.write(model_yaml)
# # serialize weights to HDF5
# model.save_weights("model_q1_2.h5")

# save history:
model_ori = open('model_q1_3.pckl', 'wb')
pickle.dump(hist.history, model_ori)
model_ori.close()

