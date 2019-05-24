import load_data
import explore_data
import vectorize_data
import build_model

import tensorflow as tf

TOP_K = 20000

def train_sequence_model(data,
                         learning_rate=1e-3,
                         epochs=1000,
                         batch_size=128,
                         blocks=2,
                         filters=64,
                         dropout_rate=0.3,
                         embedding_dim=200,
                         kernel_size=3,
                         pool_size=3):
    (train_texts, train_labels), (val_texts, val_labels) = data
    num_classes = explore_data.get_num_classes(train_labels)
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))

    x_train, x_val, word_index = vectorize_data.sequence_vectorize(
            train_texts, val_texts)
    num_features = min(len(word_index) + 1, TOP_K)

    model = build_model.sepcnn_model(blocks=blocks,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     embedding_dim=embedding_dim,
                                     dropout_rate=dropout_rate,
                                     pool_size=pool_size,
                                     input_shape=x_train.shape[1:],
                                     num_classes=num_classes,
                                     num_features=num_features)

    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=2,
            batch_size=batch_size)

    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    model.save('rotten_tomatoes_sepcnn_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]

if __name__ == '__main__':
    class_names, data = load_data.load_cook_train_data(isLemmatize=True)
    print(class_names)
    train_sequence_model(data)