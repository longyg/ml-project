from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing import sequence

import numpy as np

NGRAM_RANGE = (1, 2)
TOP_K = 20000
TOKEN_MODE = 'word'
MIN_DOCUMENT_FREQUENCY = 2
MAX_SEQUENCE_LENGTH = 500


def tfidf_vectorize(train_texts, train_labels, val_texts, ngram_range=NGRAM_RANGE, min_df=MIN_DOCUMENT_FREQUENCY):
    kwargs = {
        'ngram_range': ngram_range,
        'dtype': np.float64,
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': TOKEN_MODE,
        'min_df': min_df,
        'sublinear_tf': True,
        'stop_words': 'english'
    }
    vectorizer = TfidfVectorizer(**kwargs)

    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)

    selector = SelectKBest(f_classif, k = min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train)
    x_val = selector.transform(x_val)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    return x_train, x_val

def sequence_vectorize(train_texts, val_texts):
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    max_len = len(max(x_train, key=len))
    if max_len > MAX_SEQUENCE_LENGTH:
        max_len = MAX_SEQUENCE_LENGTH
    
    x_train = sequence.pad_sequences(x_train, maxlen = max_len)
    x_val = sequence.pad_sequences(x_val, maxlen=max_len)
    return x_train, x_val, tokenizer.word_index
