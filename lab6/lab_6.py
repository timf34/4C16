#TODO: add F1 score accuracy metric

from keras.layers import Dense, Input, Dropout, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Sequential, Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import random

DATA_PATH: str = "./dataset"


def load_data():
    X, y = [], []

    for _type in ['benign', 'malignant', 'normal']:
        X_type = np.load(f'{DATA_PATH}/{_type}/input.npy')
        y_type = np.load(f'{DATA_PATH}/{_type}/target.npy')

        # Assuming that the shapes of X_type and y_type are compatible for concatenation
        if len(X) == 0:
            X, y = X_type, y_type
        else:
            X = np.concatenate((X, X_type), axis=0)
            y = np.concatenate((y, y_type), axis=0)

    return X, y


def conv_block(input_tensor, num_filters):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    return x


def encoder_block(input_tensor, num_filters):
    """Function to add 2 convolutional layers with the parameters passed to it and then perform max pooling"""
    x = conv_block(input_tensor, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p


def decoder_block(input_tensor, concat_tensor, num_filters):
    """Function to perform up-convolution, concatenate it with the corresponding encoder block output and then add 2 convolutional layers"""
    x = UpSampling2D((2, 2))(input_tensor)
    x = concatenate([x, concat_tensor], axis=-1)
    x = conv_block(x, num_filters)
    return x


def create_model(input_shape):
    inputs = Input(input_shape)

    # Encoder
    x1, p1 = encoder_block(inputs, 64)
    x2, p2 = encoder_block(p1, 128)
    x3, p3 = encoder_block(p2, 256)
    x4, p4 = encoder_block(p3, 512)

    # Bridge
    b = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b, x4, 512)
    d2 = decoder_block(d1, x3, 256)
    d3 = decoder_block(d2, x2, 128)
    d4 = decoder_block(d3, x1, 64)

    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(d4)

    return Model(inputs=[inputs], outputs=[outputs])


def precision(y_true, y_pred):
    """Precision metric."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1_score(y_true, y_pred):
    """Calculate F1 score."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r+K.epsilon()))


def augment_data(X_train, y_train, batch_size):
    # Create two separate instances of ImageDataGenerator.
    # One for the images and one for the masks.
    data_gen_args = dict(
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed to both generators to ensure the transformations for images and masks are the same
    seed = 1
    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)

    # Combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    return train_generator


def main():
    # Load the data
    X, y = load_data()  # X and y shape: (701, 128, 128, 3)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the model
    input_shape = X_train.shape[1:]  # This gets the shape of the input data
    model = create_model(input_shape)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', precision, recall, f1_score])

    # Setup callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

    # Data augmentation
    train_generator = augment_data(X_train, y_train, batch_size=32)

    # Fit the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // 32,  # Number of batches per epoch
        validation_data=(X_val, y_val),
        epochs=50,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
