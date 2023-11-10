from keras.layers import Dense, Flatten, Dropout, MaxPool2D, Conv2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import random

DATA_PATH: str = "./dataset"


class CustomModel:
    def __init__(self):
        self.model = self.create_model()

    @staticmethod
    def create_model():
        model = Sequential()

        # First convolutional block
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # Second convolutional block
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # Third convolutional block
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # Flatten and dense layers
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))

        return model

    def train(self, train_data, train_labels, epochs=10, batch_size=32):
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Training loop
        self.model.fit(datagen.flow(train_data, train_labels, batch_size=batch_size),
                       steps_per_epoch=len(train_data) / batch_size, epochs=epochs)


class DataVisualizer:
    @staticmethod
    def visualize_data():
        for _type in ['benign', 'malignant', 'normal']:
            X = np.load(f'{DATA_PATH}/{_type}/input.npy')
            y = np.load(f'{DATA_PATH}/{_type}/target.npy')
            randomExample = random.randint(0, X.shape[0] - 1)
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(X[randomExample])
            axs[0].title.set_text('Input')
            axs[1].imshow(y[randomExample])
            axs[1].title.set_text('Output')
            fig.suptitle(_type.upper())
            plt.subplots_adjust(top=1.1)
            plt.show()


def main():
    DataVisualizer.visualize_data()
    model = CustomModel()
    model.train(np.random.rand(100, 256, 256, 3), np.random.rand(100), epochs=5)


if __name__ == '__main__':
    main()
