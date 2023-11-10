from keras.layers import Dense, Input, Dropout, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
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



def main():
    # Load the data
    X, y = load_data()  # X and y shape: (701, 128, 128, 3)


if __name__ == '__main__':
    main()
