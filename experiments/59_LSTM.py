import numpy
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from matplotlib import pyplot as plt
from keras import optimizers
from sklearn.metrics import classification_report
from keras.models import Model
from keras.layers import Input, Dense, Embedding, concatenate
from keras.layers import Dense, SimpleRNN, GRU, LSTM, Embedding
from keras.layers import CuDNNGRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Sequential
from keras.preprocessing import text, sequence
from sklearn.metrics import f1_score

train = pd.read_csv('train_75.csv')
test = pd.read_csv('valid_25.csv')
maxlen = 150
max_features = 50000
embedding_size=100


X_train = train["question_text"].fillna("dieter").values
X_test = test["question_text"].fillna("dieter").values
y_train= train["target"]
y_test=test["target"]

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
'''train = pd.concat(X_train,X_test)
test  = pd.concat(y_train,y_test)
train.to_csv("train.csv", index=False)
test.to_csv("test.csv",index=False)'''
'''
model=Sequential()
model.add(Embedding(input_dim = max_features, input_length = maxlen, output_dim = embedding_size))
model.add(Dense(20,))
model.add(Dense(1, activation='sigmoid')) 
sg=optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
'''
model = Sequential() # Call Sequential to initialize a network
model.add(Embedding(input_dim = max_features, input_length = maxlen, output_dim = embedding_size)) # Add an embedding layer which represents each unique token as a vector
model.add(LSTM(10, return_sequences=False)) # Add an LSTM layer ( will not return output at each step)
model.add(Dense(1, activation='sigmoid')) 
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

#cw={0: 0.532975   ,1: 8.08150049}

print model.summary()
# Fit the model
print "3"
model.fit(X_train, y_train, epochs=1,batch_size=1000,class_weight=cw)
# evaluate the model
print "4"
y_pred=model.predict_classes(X_test)
print f1_score(y_test,y_pred)

