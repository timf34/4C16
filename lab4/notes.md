Data augmentation works, but it takes forever to train, and you'd need to train for more than 40 epochs to make it 
performant enough to pass the tests.

Here's the code nonetheless:
```python
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
# Find out how people got it working without augmentation!

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


# we create a callback function to plot our loss function and accuracy
pltCallBack = PlotLossAccuracy()

# Fit the data augmentation generator to the training data
datagen.fit(X_train)

# Train the model using the data augmentation generator
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=1024),
                    epochs=40,
                    validation_data=(X_validation, Y_validation),
                    callbacks=[pltCallBack])

# # and train
# model.fit(X_train, Y_train,
#           batch_size=1024, epochs=40,
#           validation_data=(X_validation, Y_validation),
#           callbacks=[pltCallBack])

# If you run this cell again, the optimisation starts where you left it.
# For instance, if you have set epochs=40 in the model.fit call,
# and that you run the cell 3 times, then you are effectively running for
# 120 iterations.

```