import numpy
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from numpy import array
from numpy import asarray
from numpy import zeros

from matplotlib import pyplot as plt
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report
from keras.models import Model
from keras.layers import Input, Dense, Embedding, concatenate, Flatten
from keras.layers import Dense, SimpleRNN, GRU, LSTM, Embedding
from keras.layers import CuDNNGRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Sequential
from keras.engine.topology import Layer
from keras.preprocessing import text, sequence
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.metrics import f1_score
from sklearn import metrics
from keras import backend as K
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
def load_embeddings(file):
    embeddings = {}
    with open(file, encoding="utf8", errors='ignore') as f:
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        embeddings = dict(get_coefs(*line.split(" ")) for (i, line) in enumerate(tqdm(f)))
        
    print('Found %s word vectors.' % len(embeddings))
    return embeddings
def create_embedding_weights(tokenizer, embeddings, dimensions):
    not_embedded = defaultdict(int)
    
    word_index = tokenizer.word_index
    words_count = min(len(word_index), MAX_WORDS)
    embeddings_matrix = np.zeros((words_count, dimensions))
    for word, i in word_index.items():
        if i >= MAX_WORDS:
            continue
        if word not in embeddings:
            continue
        embedding_vector = embeddings[word]
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
            
    return embeddings_matrix
train = pd.read_csv('train_75.csv')
test = pd.read_csv('valid_25.csv')
maxlen = 150
max_features = 90000
embedding_size=100


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
f = open('/home/lakshminarasimhan/CMPS290K/experiments/GoogleNews-vectors-negative300.txt')
pretrained_embeddings=load_embeddings(f)
embedding_matrix = zeros((vocab_size, 300))
for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
inp = Input(shape=(maxlen,))
x=Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxlen, trainable=False)(inp)
#x=(Flatten())(x)
#x = Dense(64, activation="relu")(x)
#x = (LSTM(64, return_sequences=True))(x)
#x = LSTM(64, return_sequences=True)(x)
x = Attention(maxlen)(x)
x = Dense(64, activation="relu")(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)


checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
model.fit(X_train, y_train,callbacks=[checkpoint], epochs=5,batch_size=1000)
# evaluate the model
print "4"
pred_glove_val_y = model.predict([X_test], batch_size=1024, verbose=1)
for thresh in numpy.arange(0.1, 0.501, 0.01):
    thresh = numpy.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_test, (pred_glove_val_y>thresh).astype(int))))
#print f1_score(y_test,y_pred)
#print (classification_report(y_test,y_pred))

