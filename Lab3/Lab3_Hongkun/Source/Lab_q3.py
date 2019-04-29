"""
3. Implement the image classification with CNN model on anyone of the following datasets

     https://www.kaggle.com/slothkong/10-monkey-species

     https://www.kaggle.com/prasunroy/natural-images

"""
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, Activation
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
import pickle
from keras.callbacks import TensorBoard
from pathlib import Path
import pandas as pd
import os
import numpy as np
import cv2
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception, InceptionV3
import matplotlib.pylab as plt

# random seed
seed = 123
numpy.random.seed(seed)

# Set paths
training_dir = Path('./q3/training/')
val_dir = Path('./q3/validation/')
labels = Path('./q3/monkey_labels.txt')

# label info
cols = ['Label', 'Latin_Name', 'Common_Name', 'Train_Images', 'Val_Images']
labels_pd = pd.read_csv("./q3/monkey_labels.txt", names=cols, skiprows=1)


# Creating pd for training dataset
train_pd = []
for folder in os.listdir(training_dir):
    train_path = training_dir / folder
    images_train = sorted(train_path.glob('*.jpg'))

    for image_name in images_train:
        train_pd.append((str(image_name), int(folder.replace("n", ""))))
train_pd = pd.DataFrame(train_pd, columns=['image', 'label'], index=None)
# shuffle the dataset
train_pd = train_pd.sample(frac=1.).reset_index(drop=True)

# Creating pd for validation dataset
val_pd = []
for folder in os.listdir(val_dir):
    val_path = val_dir / folder
    images_val = sorted(val_path.glob('*.jpg'))
    for image_name in images_val:
        val_pd.append((str(image_name), int(folder.replace("n", ""))))

val_pd = pd.DataFrame(val_pd, columns=['image', 'label'], index=None)
# shuffle the dataset
val_pd = val_pd.sample(frac=1.).reset_index(drop=True)

# =======================================================================

print("Number of traininng samples: ", len(train_pd))
print("Number of validation samples: ", len(val_pd))

image_height = 150
image_width = 150
channels = 3
batch_size = 32
seed = 777

# Training generator
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(training_dir,
                                                    target_size=(image_height, image_width),
                                                    batch_size=batch_size,
                                                    seed=seed,
                                                    shuffle=True,
                                                    class_mode='categorical')

# Val generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(val_dir,
                                                  target_size=(image_height, image_width),
                                                  batch_size=batch_size,
                                                  seed=seed,
                                                  shuffle=False,
                                                  class_mode='categorical')
train_num = train_generator.samples
val_num = test_generator.samples


# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

# Compile model
epochs = 30
# lrate = 0.01
# decay = lrate/epochs
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# J:\5590Dl\Lab3\max_version\q3_logs
tensorboard = TensorBoard(log_dir='./q3_logs', histogram_freq=0,
                          write_graph=True, write_images=False)

# Fit the model
hist = model.fit_generator(train_generator,
                           steps_per_epoch=train_num // batch_size,
                           validation_data=test_generator,
                           validation_steps=val_num // batch_size,
                           epochs=epochs,
                           verbose=1,
                           callbacks=[tensorboard])

# acc = hist.history['acc']
# val_acc = hist.history['val_acc']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# epochs = range(1, len(acc) + 1)
#
# plt.title('Training and validation accuracy')
# plt.plot(epochs, acc, 'red', label='Training acc')
# plt.plot(epochs, val_acc, 'blue', label='Validation acc')
# plt.legend()
#
# plt.figure()
# plt.title('Training and validation loss')
# plt.plot(epochs, loss, 'red', label='Training loss')
# plt.plot(epochs, val_loss, 'blue', label='Validation loss')
#
# plt.legend()
# plt.show()






