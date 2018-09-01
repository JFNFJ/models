import os
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer

import data_helpers

# ---------------------- Initialize numpy ----------------------
np.random.seed(0)


# ---------------------- Model ----------------------
#

class Model:
    def __init__(self, weights_path, verbose=0):
        self.model = None
        self.verbose = verbose
        self.weights_path = weights_path
        self.callbacks = self.__build_model_callbacks()

    def build(self, max_words, vocabulary_inv, learning_rate=0.001):
        self.model = Sequential()
        self.model.add(Embedding(len(vocabulary_inv) + 1, 3, input_length=max_words))
        self.model.add(Dropout(0.5))
        self.model.add(Conv1D(64,
                              5,
                              padding='valid',
                              activation='relu',
                              strides=1))
        self.model.add(MaxPooling1D(pool_size=4))
        self.model.add(LSTM(50))
        self.model.add(Dense(3, activation='softmax'))
        self.model.compile(loss=categorical_crossentropy, optimizer=RMSprop(lr=learning_rate),
                           metrics=["accuracy", "mean_squared_error"])
        if self.verbose:
            print("\n\nModel summary")
            self.model.summary()

    def __build_model_callbacks(self):
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=self.verbose, mode='auto')
        checkpoint = ModelCheckpoint(filepath=self.weights_path, verbose=self.verbose, save_best_only=True)
        return [monitor, checkpoint]

    def train(self, x, y, validation_data=None, batch_size=128, epochs=60, callbacks=None):
        if not self.model:
            raise AttributeError('Model must be built first')
        return self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=self.verbose, callbacks=callbacks,
                              validation_data=validation_data)

    def evaluate(self, x, y, batch_size):
        return self.model.evaluate(x, y, batch_size=batch_size)

    def predict(self, max_words, phrase):
        tk = Tokenizer(num_words=max_words, lower=True, split=" ")
        tk.fit_on_texts(phrase)
        tokens = tk.texts_to_matrix([phrase])
        return self.model.predict(np.array(tokens))

    def load_weights(self):
        self.model.load_weights(self.weights_path)


# ---------------------- NN ----------------------

class NN:
    SEQUENCE_LENGTH = 280
    MAX_WORDS_LENGTH = SEQUENCE_LENGTH

    def __init__(self, verbose=1, weights_path='/tmp/best_weights.hdf5'):
        self.verbose = verbose

        # Model Hyperparameters
        self.training_percentage = 0.8

        # Model data
        if self.verbose:
            print("Loading data...")
        self.x_train, self.y_train, self.x_test, self.y_test, self.vocabulary_inv = load_data(self.training_percentage,
                                                                                              verbose_level=self.verbose)
        # Model itself
        self.model = Model(weights_path, verbose=self.verbose)

    def build(self):
        if self.verbose:
            print("Creating model...")
        max_words = len(self.x_train[0])
        self.model.build(max_words, self.vocabulary_inv, learning_rate=1e-2)

    def train(self):
        self.model.train(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test),
                         callbacks=self.model.callbacks)

    def score(self):
        score = self.model.evaluate(self.x_test, self.y_test, batch_size=128)
        print("SCORE: %s" % str(score))
        return

    def predict(self, phrase):
        print(self.model.predict(len(self.x_train[0]), phrase))

    def load_weights(self):
        self.model.load_weights()


# ---------------------- Categories ----------------------
#

"""
    Using one hot encoding, data will be labeles as:
        - positive: [1, 0, 0]
        - negative: [0, 0, 1]
        - neutral:: [0, 1, 0]
    Unknown labels should be ignored
"""


# ---------------------- Data Preparation ----------------------
#

def load_data(training_percentage, verbose_level=0):
    x, y, vocabulary, vocabulary_inv_list = data_helpers.load_data("./data/es/dataset")
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}

    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    train_len = int(len(x) * training_percentage)
    # x_train = sequence.pad_sequences(x[:train_len], maxlen=sequence_length, padding="post", truncating="post")
    x_train = x[:train_len]
    y_train = y[:train_len, :]
    # x_test = sequence.pad_sequences(x[train_len:], maxlen=sequence_length, padding="post", truncating="post")
    x_test = x[train_len:]
    y_test = y[train_len:, :]

    vocabulary_inv[0] = "<PAD/>"
    if verbose_level:
        print("x_train shape: %s" % str(x_train.shape))
        print("y_train shape: %s" % str(y_train.shape))
        print("x_test shape: %s" % str(x_test.shape))
        print("y_test shape: %s" % str(y_test.shape))
        print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))
    return x_train, y_train, x_test, y_test, vocabulary_inv


# ---------------------- training and scoring ----------------------
#

weight_file = '/tmp/best_weights.hdf5'
phrase = "the plot is paper-thin and the characters aren't interesting enough to watch them go about their daily activities for two whole hours . "
if not os.path.exists(weight_file):
    nn_saved = NN(weights_path=weight_file)
    nn_saved.build()
    nn_saved.train()
    nn_saved.score()

    # Predicting value
    nn_saved.predict(phrase)

nn = NN(verbose=0, weights_path=weight_file)
nn.build()
nn.load_weights()
nn.predict(phrase)
