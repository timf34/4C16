# the data structure is a tensor, ie. it is a multidimensional array
# each layer instance is callable on a tensor, and returns a tensor

# The model below contains 2 hidden layers with 30 nodes each.
# The activation functions for these 2 layers is the ReLU
# The network ends with a 10 nodes layer with softmax activation
# The first 2 hidden layers transform the original features into
# a new feature vector of size 30.
# The last layer essentially does the classification using multonomial regression
# based on these new features.


# Notes: kept overfitting on the data, and spent ages trying different combos.
# Adding data augmentation got it working asap!
# Find out how people got it working without augmentation otherwise.

from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Input, LeakyReLU
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator

# L2 regularization strength
l2_strength = 1e-4

inputs = keras.layers.Input(shape=(32, 32, 3))

x = Flatten()(inputs)

x = Dense(1024)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(0.2)(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(0.2)(x)

x = Dense(32)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(0.2)(x)

predictions = Dense(10, activation='softmax')(x)

# Create the model
model = keras.models.Model(inputs=inputs, outputs=predictions)

# Use Adam optimizer
opt = Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,   # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,   # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
)

if (model.count_params() > 5000000):
    raise Exception("Your model is unecessarily complex, scale down!")
