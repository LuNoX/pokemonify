import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print(f"Is TF built with cuda?: {tf.test.is_built_with_cuda()}")
print(f"Physical Devices: {tf.config.list_physical_devices()}")

seed = 21

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255.0
x_test = x_test/255.0

y_train = np_utils.to_categorical(y_train)
y_test  = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), input_shape=x_train.shape[1:], activation="relu", padding="same"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(class_num, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])

print(model.summary())

np.random.seed(seed)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25, batch_size=64)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

pd.DataFrame(history.history).plot()
plt.show()
