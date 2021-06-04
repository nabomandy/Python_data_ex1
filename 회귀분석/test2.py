from tensorflow.keras import layers, models
from sklearn import preprocessing
import numpy as np

def ANN_seq_func(Nin, Nh, Nout): # 13, 5, 1
    model = models.Sequential()
    """ Keras 모델 시작 """
    
    model.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,)))
    """ 입력 계층 노드 수 Nin개. 은닉 계층의 노드 수 Nh 개, 활성함수는 relu """
    
    model.add(layers.Dense(Nout, activation='relu'))
    """ 출력 노드 수 Nout 개, 활성함수는 relu """
    
    model.compile(loss='mse', optimizer='sgd')
    """ cost 함수 - mse - 평균 제곱 오차 최적화 알고리즘 -SGD(확률적 경사하강법) """
    
    return model



from tensorflow.keras import datasets

def Data_func():
    (X_train, y_train), (X_test, y_test) = datasets.boston_housing.load_data()
    scaler =preprocessing.MinMaxScaler() # 데이터 정규하
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print (X_train.shape, y_train.shape)
    
    return (X_train, y_train), (X_test, y_test)

def main():
    Nin = 13
    Nh = 5
    Nout = 1
    
    model = ANN_seq_func(Nin, Nh, Nout)
    (X_train, y_train), (X_test, y_test) = Data_func()
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=100, validation_split=0.2, verbose=2)
    
    performance_test = model.evaluate(X_test, y_test, batch_size=100)
    print('\nTest Loss -> {:.2f}'.format(performance_test))
    
    history = history.history
    
    """Cost/Loss 변화 추이 그래프"""
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Verification'], loc=0)
    plt.show()

    