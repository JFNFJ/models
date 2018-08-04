"""
Train convolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf
For "CNN-rand" and "CNN-non-static" gets to 88-90%, and "CNN-static" - 85% after 2-5 epochs with following settings:
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50
Differences from original article:
- larger IMDB corpus, longer sentences; sentence length is very important, just like data size
- smaller embedding dimension, 50 instead of 300
- 2 filter sizes instead of original 3
- fewer filters; original work uses 100, experiments show that 3-10 is enough;
- random initialization is no worse than word2vec init on IMDB corpus
- sliding Max Pooling instead of original Global Pooling
"""

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import imdb
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from sklearn import preprocessing
from keras.losses import categorical_crossentropy

import data_helpers

# ---------------------- Initialize numpy ----------------------
np.random.seed(0)


# ---------------------- Model ----------------------
#
class Model:
    def __init__(self, verbose=0):
        self.model = None
        self.verbose = verbose
        self.callbacks = self.__build_model_callbacks()

    def build(self, dropout=(0.5, 0.5, 0.8), learning_rate=0.01):
        self.model = Sequential()
        self.model.add(Dense(1024, input_shape=(max_words,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout[0]))
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout[1]))
        self.model.add(Dense(256))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dropout(dropout[2]))
        self.model.add(Dense(len(categories_labels)))
        self.model.add(Activation('softmax'))
        self.model.compile(loss=categorical_crossentropy, optimizer=RMSprop(lr=learning_rate), metrics=["accuracy"])
        if self.verbose:
            print("\n\nModel summary")
            self.model.summary()

    # Model callback
    def __build_model_callbacks(self):
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=self.verbose, mode='auto')
        checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', verbose=self.verbose, save_best_only=True)
        return [monitor, checkpoint]

    def train(self, x, y, validation_data=None, batch_size=128, epochs=60, callbacks=None):
        print("x shape: %s" % str(x.shape))
        print("y shape: %s" % str(y.shape))
        if not self.model:
            raise AttributeError('Model must be built first')
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=self.verbose, callbacks=callbacks,
                       validation_data=validation_data)


# ---------------------- Categories ----------------------
#

"""
    Using one hot encoding, data will be labeles as:
        - positive: [1, 0, 0]
        - negative: [0, 0, 1]
        - neutral:: [0, 1, 0]
    Unknown labels should be ignored
"""

categories_labels = ['positive', 'neutral', 'negative']
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(categories_labels)

# ---------------------- Parameters section -------------------
#
# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
# model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static
verbose = 1

# Data source
data_source = "local_dir"  # keras_data_set|local_dir

# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50
TRAINING_PERCENTAGE = 0.8

# Training parameters
batch_size = 64
num_epochs = 20

# Prepossessing parameters
sequence_length = 280
max_words = sequence_length

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10


# ---------------------- Data Preparation ----------------------
#

def load_data(data_source, _verbose=0):
    assert data_source in ["keras_data_set", "local_dir"], "Unknown data source"
    if data_source == "keras_data_set":
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words, start_char=None,
                                                              oov_char=None, index_from=None)

        x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
        x_test = sequence.pad_sequences(x_test, maxlen=sequence_length, padding="post", truncating="post")

        vocabulary = imdb.get_word_index()
        vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
    else:
        x, y, vocabulary, vocabulary_inv_list = data_helpers.load_data()
        vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}

        # Shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x = x[shuffle_indices]
        y = y[shuffle_indices]
        train_len = int(len(x) * TRAINING_PERCENTAGE)
        x_train = sequence.pad_sequences(x[:train_len], maxlen=sequence_length, padding="post", truncating="post")
        y_train = y[:train_len, :]
        x_test = sequence.pad_sequences(x[train_len:], maxlen=sequence_length, padding="post", truncating="post")
        y_test = y[train_len:, :]

    vocabulary_inv[0] = "<PAD/>"
    if _verbose:
        print("x_train shape: %s" % str(x_train.shape))
        print("y_train shape: %s" % str(y_train.shape))
        print("x_test shape: %s" % str(x_test.shape))
        print("y_test shape: %s" % str(y_test.shape))
        print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))
    return x_train, y_train, x_test, y_test, vocabulary_inv


print("Loading data...")
x_train, y_train, x_test, y_test, vocabulary_inv = load_data(data_source, _verbose=verbose)

print("Training samples size:" + str(len(x_train)))
print("Training labels size:" + str(len(y_train)))

model = Model(verbose=verbose)
model.build()
model.train(x_train, y_train, validation_data=(x_test, y_test), callbacks=model.callbacks)
