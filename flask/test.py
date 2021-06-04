import tensorflow as tp
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# predict model 만들기
number_mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = number_mnist.load_data()

# print(test_images[300])
model = keras.Sequential([ keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(128, activation='relu'),
                          keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)