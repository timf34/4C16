# the data structure is a tensor, ie. it is a multidimensional array
# each layer instance is callable on a tensor, and returns a tensor

# The model below contains 2 hidden layers with 30 nodes each.
# The activation functions for these 2 layers is the ReLU
# The network ends with a 10 nodes layer with softmax activation
# The first 2 hidden layers transform the original features into
# a new feature vector of size 30.
# The last layer essentially does the classification using multonomial regression
# based on these new features.

from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Input, LeakyReLU
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator

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

if (model.count_params() > 5000000):
    raise Exception("Your model is unecessarily complex, scale down!")
