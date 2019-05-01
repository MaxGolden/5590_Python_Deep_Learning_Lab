
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
np.random.seed(0)

#load data
train_df = pd.read_csv('train.tsv', sep='\t', header=0)
#test_df = pd.read_csv('test.tsv', sep='\t', header=0)

all_data = train_df.drop(['SentenceId', 'PhraseId', 'Sentiment'], axis=1)
all_class = train_df['Sentiment']

#prepare data
all_data['Phrase'] = all_data['Phrase'].apply(lambda x:x.lower())
all_data['Phrase'] = all_data['Phrase'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(all_data['Phrase'].values)
X = tokenizer.texts_to_sequences(all_data['Phrase'].values)
X = pad_sequences(X)

#encode classes
labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(train_df['Sentiment'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

embed_dim = 128
#CNN model
model_CNN = Sequential()
model_CNN.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
model_CNN.add(Dropout(0.2))
model_CNN.add(Conv1D(64, 5, activation='relu'))
model_CNN.add(MaxPooling1D(pool_size=4))
model_CNN.add(Flatten())
model_CNN.add(Dense(5, activation='softmax'))
model_CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

tensorboard_CNN = TensorBoard(log_dir='./MovieReview_resultsCNN', histogram_freq=0,
                          write_graph=True, write_images=False)
batch_size = 32
histCNN = model_CNN.fit(X_train, Y_train,
                 epochs=1,
                 batch_size=batch_size,
                 verbose=2,
                 callbacks=[tensorboard_CNN],
                 validation_data=(X_test,Y_test))

lstm_out = 196
#LSTM model
model_LSTM = Sequential()
model_LSTM.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
model_LSTM.add(SpatialDropout1D(0.4))
model_LSTM.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model_LSTM.add(Dense(5, activation='softmax'))
model_LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

tensorboard_LSTM = TensorBoard(log_dir='./MovieReview_resultsLSTM', histogram_freq=0,
                          write_graph=True, write_images=False)

batch_size = 32
histLSTM = model_LSTM.fit(X_train, Y_train,
                 epochs=1,
                 batch_size=batch_size,
                 verbose=2,
                 callbacks=[tensorboard_LSTM],
                 validation_data=(X_test,Y_test))



