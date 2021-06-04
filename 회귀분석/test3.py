# tensoorflow와 tf.keras를 임포트 합니다.
import tensorflow as tf
from tensorflow import keras

# 헬퍼(helper) 라이브러리를 임포트 합ㄴ디ㅏ.
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

print()
print("================")
print()

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = [ 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape # (60000,28,28)

train_labels.shape # (60000,)

train_labels

len(train_labels) # 60000

len(test_labels) # 10000

plt.figure()
plt.imshow(train_images[9])
plt.colorbar()
plt.grid(False)
plt.show()
print(class_names[train_labels[9]])

train_images[3]

# 값이 커지면 발산하게된다. 그래서 0 -1 사이에 올 수 있게 아래의 작업을 실행한다
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images[3]

model = keras.Sequential([ keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(128, activation='relu'),
                          keras.layers.Dense(10, activation='softmax')])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\n테스트 정확도:', test_acc)

predictions = model.predict(test_images[[200]]) # 테스트 이미지 중 첫번째 거 꺼내기
print(np.argmax(predictions)) # 1

plt.figure()
plt.imshow(train_images[200])
plt.colorbar()
plt.grid(False)
plt.show()
print(class_names[test_labels[200]])
print(test_labels[200])

test_images.shape # (10000, 28, 28)

