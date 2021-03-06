import explore_data
import build_model
import load_data
import tensorflow as tf

def train_embed_model(data,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.5):
    (train_texts, train_labels), (val_texts, val_labels) = data

    num_classes = explore_data.get_num_classes(train_labels)

    model = build_model.embedding_model(layers, units, num_classes, dropout_rate)

    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    training_dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(train_texts, tf.string),
            tf.cast(train_labels, tf.int32)
        )
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(val_texts, tf.string),
            tf.cast(val_labels, tf.int32)
        )
    )

    history = model.fit(training_dataset.shuffle(1000).batch(batch_size),
                        epochs=epochs,
#                         callbacks=callbacks,
                        validation_data=validation_dataset.batch(batch_size),
                        verbose=1)
    
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
        acc=history['val_acc'][-1], loss=history['val_loss'][-1]))
    
#     return history['val_acc'][-1], history['val_loss'][-1]
    return history

if __name__ == '__main__':
    class_names, data = load_data.load_cook_train_data(isLemmatize=True)
    print(class_names)
    train_embed_model(data, layers=1, units=32)