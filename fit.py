from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np


MAX_LEN = 500  # max number of words in essay to use


def get_model(embedding_matrix, learning_rate=0.001):
    optimizer = Adam(learning_rate=learning_rate)

    nb_words, embed_size = embedding_matrix.shape
    inp = Input(shape=(MAX_LEN,))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(LSTM(20, return_sequences=True, dropout=0.4, recurrent_dropout=0.2))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(5, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy',
                                                                            #tf.keras.metrics.AUC(),
                                                                            tf.keras.metrics.Precision(),
                                                                            tf.keras.metrics.Recall()])
    return model


def fit_model(model, X_train, y_train, model_name, patience=10, validation_split=0.15, epochs=50, batch_size=32):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    checkpoints = tf.keras.callbacks.ModelCheckpoint(f'models/model-{model_name}' +
                                                     '-Epoch-{epoch:02d}-Loss-{val_loss:.4f}-Acc-{val_accuracy:.4f}.h5',
                                                     save_best_only=True)
    callbacks = [early_stopping, checkpoints]
    print(model.summary())
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                        callbacks=callbacks, workers=-1, use_multiprocessing=True)
    return history


def read_input():
    embedding_matrix = np.load('data/embedding_matrix.npy')
    X, y = np.load('data/train_input.npy'), np.load('data/train_output.npy')
    return X, y, embedding_matrix


if __name__ == '__main__':
    X, y, embedding_matrix = read_input()
    for i in range(y.shape[1]):
        model = get_model(embedding_matrix)
        fit_model(model, X, y[:, i], i)
        print(f"Model for {i} class has been fitted.")
