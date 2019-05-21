import tensorflow as tf
import load_data
import explore_data
import vectorize_data
import build_model
import nltk

def train_mlp_model(data,
                    learning_rate=1e-3,
                    epochs=100,
                    batch_size=128,
                    layers=2,
                    units=64,
                    dropout_rate=0.2):
    (train_texts, train_labels), (val_texts, val_labels) = data
    
    num_classes = explore_data.get_num_classes(train_labels)

    unexpected_labels = [i for i in val_labels if i not in range(num_classes)]
    if len(unexpected_labels) > 0:
        raise ValueError('Unexpected label values found in validation set: '
                         '{unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(unexpected_labels=unexpected_labels))
       
    x_train, x_val = vectorize_data.tfidf_vectorize(train_texts, train_labels, val_texts)

    model = build_model.mlp_model(layers=layers,
                                  units=units,
                                  dropout_rate=dropout_rate,
                                  input_shape=x_train.shape[1:],
                                  num_classes=num_classes)
    
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    history = model.fit(x_train,
                        train_labels,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(x_val, val_labels),
                        verbose=2,
                        batch_size=batch_size)
    
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
        acc=history['val_acc'][-1], loss=history['val_loss'][-1]))
    
    model.save('mlp_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]

if __name__ == '__main__':
    nltk.download('wordnet')
    class_names, data = load_data.load_cook_train_data(isLemmatize=True)
    print(class_names)
    train_mlp_model(data)

