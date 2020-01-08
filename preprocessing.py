import numpy as np
import pandas as pd
import sys
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
import json

nltk.download('punkt')
nltk.download('stopwords')

CLASSES = ('cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN')
EMBED_SIZE = 300  # how big is each word vector
MAX_FEATURES = 31000  # how many unique words to use (i.e num rows in embedding vector)
MAX_LEN = 500  # max number of words in essay to use
EMBEDDING_FILE = 'data/glove.840B.300d.txt'


def to_binary(df, fields=CLASSES):
    df = df.copy()
    for field in fields:
        df[field] = (df[field] == 'y').astype(np.int16)
    return df


def filter_no_emotional(df):
    df = df.copy()
    filter_field = pd.Series(np.array([True] * df.shape[0]))
    for field in CLASSES:
        filter_field &= df[field] == 0
    return df[~filter_field].reset_index(drop=True)


def read_df(path):
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df = df.drop(columns='#AUTHID')
    return df


def clean_str(string, TREC=False):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s ", string)
    string = re.sub(r"\'ve", " have ", string)
    string = re.sub(r"n\'t", " not ", string)
    string = re.sub(r"\'re", " are ", string)
    string = re.sub(r"\'d" , " would ", string)
    string = re.sub(r"\'ll", " will ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def filter_stop_words(sentences):
    stop_words = set(stopwords.words('english'))
    arr = sentences.copy()
    for i, s in enumerate(arr):
        words = nltk.word_tokenize(s)
        s = ''
        for word in words:
            if word not in stop_words:
                s += word + ' '
        arr[i] = s
    return arr


def early_preprocessing(df):
    df = to_binary(df)
    df = filter_no_emotional(df)
    X = df.TEXT.values
    y = df[[*CLASSES]].values
    X = np.vectorize(clean_str)(X)
    X = filter_stop_words(X)
    return X, y


def tokenizing_and_filter_outliers(X, y):
    tokenizer = Tokenizer(num_words=MAX_FEATURES, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789')
    tokenizer.fit_on_texts(X)
    list_tokenized_train = tokenizer.texts_to_sequences(X)
    number_of_words_in_sentence = np.array(list(map(len, list_tokenized_train)))
    X_sequence = pad_sequences(list_tokenized_train, maxlen=MAX_LEN)

    outliers_greater = number_of_words_in_sentence > MAX_LEN
    outliers_less = number_of_words_in_sentence < 150
    outliers = outliers_greater | outliers_less

    X_sequence = X_sequence[~outliers]
    y = y[~outliers]

    word_index = tokenizer.word_index
    with open('data/word_index.json', 'w') as file:
        json.dump(word_index, file, ensure_ascii=False)

    return X_sequence, y, word_index


def read_embedding():
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = {}
    invalid = []

    for o in open(EMBEDDING_FILE, encoding='utf-8'):
        word = emb = None
        try:
            word, emb = get_coefs(*o.strip().split())
        except Exception as e:
            invalid.append(o)
        if emb is not None:
            if emb.shape[0] != EMBED_SIZE:
                invalid.append(o)
            else:
                embeddings_index[word] = emb

    print(f"Invalid embedding strings: {len(invalid)}")
    return embeddings_index


def create_embedding_matrix(word_index):
    embeddings_index = read_embedding()

    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    no_known = {}

    nb_words = min(MAX_FEATURES, len(word_index) + 1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBED_SIZE))

    for word, i in word_index.items():
        if i >= MAX_FEATURES:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            word = word.replace("'", "")
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                no_known[i] = word

    print(f"There are {len(no_known)} unknown words")
    np.save('data/embedding_matrix', embedding_matrix)
    print(f"Embedding matrix has been saved.")
    return embedding_matrix


def preprocessing(input_file):
    df = read_df(input_file)
    X, y = early_preprocessing(df)
    X, y, word_index = tokenizing_and_filter_outliers(X, y)
    embedding_matrix = create_embedding_matrix(word_index)
    np.save('data/train_input', X)
    np.save('data/train_output', y)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("Need input data file")
    preprocessing(sys.argv[1])
