"""
요즘은 ann보다는 cnn 방식을 사용한다고 한다
"""

import tensorflow as tf
import keras
import numpy

x = numpy.array([0,1,2,3,4])
y = x * 2 + 1

# Keras모델 시작
model = keras.models.Sequential()

# 인공지능 계층을 하나 추가 (입력 노드 하나와 가중치 하나)
model.add(keras.layers.Dense(1, input_shape=(1,)))

# SGD - 확률적 경사하강법, mse - 평균 제곱 오차 cost function
model.compile('SGD', 'mse')


# 지도 학습 # 1000에포크(학습횟수), verbose는 학습 진행 사항 로그 출력여부, 0은 표시하지 않음
model.fit(x[:2], y[:2], epochs=1000, verbose=0)

print('Targets:', y[2:]) # 실제 결과  
print('Predictions:', model.predict(x[2:]).flatten()) # flatten은 결과를 그냥 한줄로 보려고 한 기능함수