import numpy as np
from keras.preprocessing.sequence import _remove_long_seq

def load_data(path='', num_words=None, skip_top=0, maxlen=None, seed=113, start_char=1, oov_char=2, index_from=3, **kwargs):
    if kwargs:
        raise TypeError('Unrecognized keyword arguemnts: ' + str(kwargs))

    with np.load(path) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    # Seed set to random
    np.random.seed(seed)

    indices = np.arange(len(x_train)) # Returns a vector of indeces x-spaced
    np.random.shuffle(indices) # Shuffle those numbers
    x_train = x_train[indices] # Reorders the training set randomly
    labels_train = labels_train[indices] # Reorders the labels with the same order as x_train, so labels matches each word

    # reorders the training set as well
    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]
    
    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. '
                             'Increase maxlen.')
    if not num_words:
        num_words = max([max(x) for x in xs])
    
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs]
    else:
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)