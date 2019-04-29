import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import pickle
from keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_yaml
import matplotlib.pyplot as plt
from keras import metrics
from keras.optimizers import SGD


f = open('model_q2_2.pckl', 'rb')
history1 = pickle.load(f)
f.close()
f1 = open('model_q2_1.pckl', 'rb')
history2 = pickle.load(f1)
f1.close()

# acc
plt.plot(history1['val_acc'])
plt.plot(history2['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['model_1', 'model_2'], loc='upper left')
plt.show()










