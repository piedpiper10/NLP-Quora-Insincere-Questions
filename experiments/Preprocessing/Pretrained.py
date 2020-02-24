import numpy
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from numpy import array
from numpy import asarray
from numpy import zeros
from collections import defaultdict
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
from collections import defaultdict
import operator
import re
from tqdm import tqdm
class Preprocessor:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def build_tf_dict(self, sentences):
        """
        Build a simple TF (term frequency) dictionary for all words in the provided sentences.
        """
        tf_dict = defaultdict(int)
        for sentence in sentences:
            for word in sentence:
                tf_dict[word] += 1
        return tf_dict

    def check_coverage(self, tf_dictionary):
        """
        Build a simple list of words that are not embedded. Can be used down the stream to preprocess them to something
        known.
        """
        in_vocabulary = defaultdict(int)
        out_of_vocabulary = defaultdict(int)
        in_count = 0
        out_count = 0

        for word in tf_dictionary:
            if word in self.embeddings:
                in_vocabulary[word] = self.embeddings[word]
                in_count += tf_dictionary[word]
            else:
                out_of_vocabulary[word] = tf_dictionary[word]
                out_count += tf_dictionary[word]
	print out_of_vocabulary
        percent_tf = len(in_vocabulary) / len(tf_dictionary)
        percent_all = in_count / (in_count + out_count)
        print('Found embeddings for {:.2%} of vocabulary and {:.2%} of all text'.format(percent_tf, percent_all))

        return sorted(out_of_vocabulary.items(), key=operator.itemgetter(1))[::-1]

    def clean_punctuation(self, text):
        result = text
        
        for punct in "/-'":
            result = result.replace(punct, ' ')
        for punct in '&':
            result = result.replace(punct, ' {punct} ')
        for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~'+'''"'"''' :
            result = result.replace(punct, '')

        return result

    def clean_digits(self, text):
        result = text
        result = re.sub('[0-9]{5,}', '#####', result)
        result = re.sub('[0-9]{4}', '####', result)
        result = re.sub('[0-9]{3}', '###', result)
        result = re.sub('[0-9]{2}', '##', result)
        return result

    def clean_misspelling(self, text):
        mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium'}

        def _get_mispell(mispell_dict):
            mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
            return mispell_dict, mispell_re

        mispellings, mispellings_re = _get_mispell(mispell_dict)

        def replace(match):
            return mispellings[match.group(0)]

        return mispellings_re.sub(replace, text)
    
    def apply_cleaning_function(self, fn, texts, description = ""):
        result = [fn(text) for text in texts]
        sentences = [text.split() for text in result]
        tf_dict = self.build_tf_dict(sentences)
        oov = self.check_coverage(tf_dict)
#         print(oov[:10])

        return result

    def preprocess_for_embeddings_coverage(self, texts):
        result = texts

        sentences = [text.split() for text in result]
        tf_dict = self.build_tf_dict(sentences)
        oov = self.check_coverage(tf_dict)

        result = self.apply_cleaning_function(lambda x: self.clean_punctuation(x), result, "Cleaning punctuation...")
#         result = self.apply_cleaning_function(lambda x: self.clean_digits(x), result, "Cleaning numbers...")
#         result = self.apply_cleaning_function(lambda x: self.clean_misspelling(x), result, "Cleaning misspelled words...")

        return result
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
    with open(file) as f:
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        embeddings = dict(get_coefs(*line.split(" ")) for (i, line) in enumerate(tqdm(f)))
        
    print('Found %s word vectors.' % len(embeddings))
    return embeddings
def create_embedding_weights(tokenizer, embeddings, dimensions):
    not_embedded = defaultdict(int)
    word_index = tokenizer.word_index
    words_count = len(word_index)+1
    embeddings_matrix = np.zeros((words_count, dimensions))
    for word, i in word_index.items():
        if i >=90000:
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
embedding_size=300


X_train = train["question_text"].fillna("dieter").values
X_test = test["question_text"].fillna("dieter").values
y_train= train["target"]
y_test=test["target"]
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

f = '/home/lakshminarasimhan/CMPS290K/experiments/GoogleNews-vectors-negative300.bin'
pretrained_embeddings = load_embeddings(f)
preprocessor = Preprocessor(pretrained_embeddings)
X_train = preprocessor.preprocess_for_embeddings_coverage(X_train)
X_test = preprocessor.preprocess_for_embeddings_coverage(X_test)
t = Tokenizer()
t.fit_on_texts(list(X_train) + list(X_test))
vocab_size = len(t.word_index) + 1
X_train = t.texts_to_sequences(X_train)
X_test = t.texts_to_sequences(X_test)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
pretrained_emb_weights = create_embedding_weights(t, pretrained_embeddings, embedding_size)
inp = Input(shape=(maxlen,))
x=Embedding(vocab_size, 300, weights=[pretrained_emb_weights], input_length=maxlen, trainable=False)(inp)
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

