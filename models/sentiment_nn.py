import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import imdb
from keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

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

    def build(self, max_words, learning_rate=0.001):
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
        checkpoint = ModelCheckpoint(filepath='/tmp/best_weights.hdf5', verbose=self.verbose, save_best_only=True)
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
        return model.model.predict(np.array(tokens))

    def load_from_saved_weights(self, weigths_path):
        self.build()
        return


# ---------------------- Categories ----------------------
#

"""
    Using one hot encoding, data will be labeles as:
        - positive: [1, 0, 0]
        - negative: [0, 0, 1]
        - neutral:: [0, 1, 0]
    Unknown labels should be ignored
"""

# ---------------------- Parameters section -------------------
#
# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
# model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static
verbose = 1

# Data source
data_source = "local_dir"  # keras_data_set|local_dir

# Model Hyperparameters
TRAINING_PERCENTAGE = 0.8

# Prepossessing parameters
SEQUENCE_LENGTH = 280
MAX_WORDS_LENGTH = SEQUENCE_LENGTH


# ---------------------- Data Preparation ----------------------
#

def load_data(data_source, verbose_level=0):
    assert data_source in ["keras_data_set", "local_dir"], "Unknown data source"
    if data_source == "keras_data_set":
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS_LENGTH, start_char=None,
                                                              oov_char=None, index_from=None)

        x_train = sequence.pad_sequences(x_train, maxlen=SEQUENCE_LENGTH, padding="post", truncating="post")
        x_test = sequence.pad_sequences(x_test, maxlen=SEQUENCE_LENGTH, padding="post", truncating="post")

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


print("Loading data...")
x_train, y_train, x_test, y_test, vocabulary_inv = load_data(data_source, verbose_level=verbose)

print("Creating model...")
max_words = x_train.shape[1]
model = Model(verbose=verbose)
model.build(max_words, learning_rate=1e-2)
model.train(x_train, y_train, validation_data=(x_test, y_test), callbacks=model.callbacks)
score = model.evaluate(x_test, y_test, batch_size=128)
print("SCORE: %s" % str(score))

#   Predicting value
#
#phrase = "emerges as one thing rare , an issue movie that's so honest and keenly observed that it does not feel like one . "
phrase = "the plot is paper-thin and the characters aren't interesting enough to watch them go about their daily activities for two whole hours . "
print(model.model.predict(max_words, phrase))
