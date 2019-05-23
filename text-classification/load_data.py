import pandas as pd
import random
import re
from nltk.stem import WordNetLemmatizer
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
data_dir_path = os.path.join(dir_path, '../cook-prediction')
print(data_dir_path)
train_data_file = os.path.realpath(os.path.join(data_dir_path, 'train.json'))
print(train_data_file)
test_data_file = os.path.realpath(os.path.join(data_dir_path, 'test.json'))
print(test_data_file)

def load_cook_train_data(validation_split=0.2, isLemmatize=False):
    return _load_cook_train_data(train_data_file, 
                                 validation_split=validation_split, 
                                 isLemmatize=isLemmatize)
def load_cook_test_data(isLemmatize=False):
    return _load_cook_test_data(test_data_file, isLemmatize=isLemmatize)

def _load_cook_train_data(data_path, validation_split=0.2, seed=42, isLemmatize=False):
    data = pd.read_json(data_path)
    class_names = list(data.cuisine.unique())

    data.ingredients = data.ingredients.apply(lambda x: _text_transform(x, isLemmatize))
    data = _filter_data(data)
    train_texts = data.ingredients.values

    train_labels = data.cuisine.apply(lambda x: class_names.index(x)).values

    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return class_names, _split_train_and_validation_data(train_texts, train_labels, validation_split)

def _load_cook_test_data(data_path, isLemmatize=False):
    data = pd.read_json(data_path)

    test_texts = data.ingredients.apply(lambda x: _text_transform(x, isLemmatize)).values

    return test_texts

def _lower_texts(data):
    data.ingredients = data.ingredients.apply(lambda x: list(map(lambda y: y.lower(), x)))
    return data

def _filter_data(data):
    data['num_ingredients'] = data.ingredients.apply(lambda x: len(x.split()))
    data = data[data['num_ingredients'] > 0]
    return data

def _text_transform(texts, isLemmatize=False):
    texts = _remove_unused_texts(texts)
    if len(texts) < 1:
        return ''

    lemmatizer = WordNetLemmatizer()
    # stop_pattern = re.compile('[\d’%]')

    string = ' '.join(texts).lower()

    for key, value in _get_replacements().items():
        string = string.replace(key, value)

    words = []
    for word in string.split():
        word = _replace_word(word)
        # not stop_pattern.match(word) and 
        if len(word) > 2:
            if isLemmatize:
                # 词形还原，比如 eggs 还原为 egg
                word = lemmatizer.lemmatize(word)
            words.append(word)
    return ' '.join(words)

def _replace_word(word):
    for key, value in _get_word_replacements().items():
        word = word.replace(key, value)
    return word

def _remove_unused_texts(texts):
    unused_keywords = _get_unused_keywords()
    texts = [text for text in texts if not _ifInList(text, unused_keywords)]
    return texts

def _ifInList(text, wordList):
    for key in wordList:
        if key.lower() in text.lower():
            return True
    return False

def _split_train_and_validation_data(texts, labels, validation_split):
    num_training_samples = int((1 - validation_split) * len(texts))

    train_texts = texts[:num_training_samples]
    train_lables = labels[:num_training_samples]

    val_texts = texts[num_training_samples:]
    val_labels = labels[num_training_samples:]
    return ((train_texts, train_lables), (val_texts, val_labels))

def _get_unused_keywords():
    return ["i can't believe"]

def _get_word_replacements():
    return {'chilies': 'chili',
            'mayonnais': 'mayonnaise',
            'hellmann\'\'s': 'hellmann',
            'hellmann\'s®': 'hellmann',
            'hellmann\'': 'hellmann',
            'rice®': 'rice'
        }

def _get_replacements():
    return {'wasabe': 'wasabi', '-': '', 'sauc': 'sauce',
            'baby spinach': 'babyspinach', 'coconut cream': 'coconutcream',
            'coriander seeds': 'corianderseeds', 'corn tortillas': 'corntortillas',
            'cream cheese': 'creamcheese', 'fish sauce': 'fishsauce',
            'purple onion': 'purpleonion','refried beans': 'refriedbeans', 
            'rice cakes': 'ricecakes', 'rice syrup': 'ricesyrup', 
            'sour cream': 'sourcream', 'toasted sesame seeds': 'toastedsesameseeds', 
            'toasted sesame oil': 'toastedsesameoil', 'yellow onion': 'yellowonion', 
            '7 up': '7up',
            'uncle ben\'s®': 'unclebens',
            'uncle ben\'s': 'unclebens',
            'uncle bens': 'unclebens'
            }