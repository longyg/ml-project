import pandas as pd
import random
from nltk.stem import WordNetLemmatizer

def load_cook_train_data(data_path, validation_split=0.2, seed=123, isLemmatize=False):
    data = pd.read_json(data_path)
    class_names = list(data.cuisine.unique())
    train_texts = data.ingredients.apply(lambda x: _text_transform(x, isLemmatize))
    train_labels = data.cuisine.apply(lambda x: class_names.index(x))

    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return class_names, _split_train_and_validation_data(train_texts, train_labels, validation_split)

def load_cook_test_data(data_path, isLemmatize=False):
    data = pd.read_json(data_path)
    test_texts = data.ingredients.apply(lambda x: _text_transform(x, isLemmatize))
    return test_texts

def _text_transform(texts, isLemmatize=False):
    lemmatizer = WordNetLemmatizer()
    string = ' '.join(texts)
    words = []
    for word in string.split():
        if isLemmatize:
            # 词形还原，比如 eggs 还原为 egg
            word = lemmatizer.lemmatize(word)
        words.append(word)
    return ' '.join(words)

def _split_train_and_validation_data(texts, labels, validation_split):
    num_training_samples = int((1 - validation_split) * len(texts))

    train_texts = texts[:num_training_samples]
    train_lables = labels[:num_training_samples]

    val_texts = texts[num_training_samples:]
    val_labels = labels[num_training_samples:]
    return ((train_texts, train_lables), (val_texts, val_labels))