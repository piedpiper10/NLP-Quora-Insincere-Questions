import numpy
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from matplotlib import pyplot as plt
from keras import optimizers
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.models import Model
from keras.layers import Input, Dense, Embedding, concatenate, Flatten,Activation, Dropout
from keras.layers import Dense, SimpleRNN, GRU, LSTM, Embedding
from keras.layers import CuDNNGRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Sequential
from keras.preprocessing import text, sequence
from sklearn.metrics import f1_score

train = pd.read_csv('train_75.csv')
test = pd.read_csv('valid_25.csv')
maxlen = 150
max_features = 50000
embedding_size=300


X_train = train["question_text"].fillna("dieter").values
X_test = test["question_text"].fillna("dieter").values
y_train= train["target"]
y_test=test["target"]

t = Tokenizer()
t.fit_on_texts(list(X_train) + list(X_test))
vocab_size = len(t.word_index) + 1
X_train = t.texts_to_sequences(X_train)
X_test = t.texts_to_sequences(X_test)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
embeddings_index = dict()
f = open('GoogleNews-vectors-negative300.txt')
for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
f.close()
embedding_matrix = zeros((vocab_size, 300))
for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector


model = Sequential() # Call Sequential to initialize a network
model.add(Embedding(input_dim = max_features, input_length = maxlen, output_dim = embedding_size))
#model.add(Flatten())
model.add(LSTM(64))
model.add(Dense(256,name='FC1'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1,name='out_layer'))
model.add(Activation('sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
model.fit(X_train, y_train, epochs=1,batch_size=1000)
# evaluate the model
print "4"
y_pred=model.predict_classes(X_test)
print f1_score(y_test,y_pred)

