# Importing libraries
from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from master_function import data_preprocessing, plot_train_test_values
from master_function import calculate_accuracy, model_bias
from sklearn.metrics import mean_squared_error

# Set the start and end dates for the data
start_date = '1990-01-01'
end_date   = '2023-06-01'

# Fetch S&P 500 price data
# S&P500 데이터를 다운로드
data = np.array((pdr.get_data_fred('SP500', start = start_date, end = end_date)).dropna())

# Difference the data and make it stationary
'''
x = [100, 102, 105, 103]
np.diff(x)  # → [2, 3, -2]
=> S&P 500의 종가를 하루 단위로 차분해서,하루하루 수익률 변화 (혹은 가격 변화량) 을 구하는 것
'''
data = np.diff(data[:, 0])

# Setting the hyperparameters
num_lags = 100
train_test_split = 0.80
num_neurons_in_hidden_layers = 20
num_epochs = 500
batch_size = 16

# Creating the training and test sets
x_train, y_train, x_test, y_test = data_preprocessing(data, num_lags, train_test_split)

# Designing the architecture of the model
# MLP 모델 구축
model = Sequential()

# First hidden layer
'''
은닉층 1: 20개 뉴런, ReLU
은닉층 2: 20개 뉴런, ReLU
출력층: 1개 뉴런 (다음 시점 수익률 예측)
'''
model.add(Dense(num_neurons_in_hidden_layers, input_dim = num_lags, activation = 'relu'))  

# Second hidden layer
model.add(Dense(num_neurons_in_hidden_layers, activation = 'relu'))  

# Output layer
model.add(Dense(1))

# Compiling
'''
손실 함수: 평균 제곱 오차(MSE)
최적화 기법: Adam
'''
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

# Fitting the model
model.fit(x_train, np.reshape(y_train, (-1, 1)), epochs = num_epochs, batch_size = batch_size)

# Predicting in-sample
y_predicted_train = np.reshape(model.predict(x_train), (-1, 1))

# Predicting out-of-sample
y_predicted = np.reshape(model.predict(x_test), (-1, 1))

# plotting
plot_train_test_values(100, 50, y_train, y_test, y_predicted)

# Performance evaluation
print('---')
print('Accuracy Train = ', round(calculate_accuracy(y_predicted_train, y_train), 2), '%')
print('Accuracy Test = ', round(calculate_accuracy(y_predicted, y_test), 2), '%')
print('RMSE Train = ', round(np.sqrt(mean_squared_error(y_predicted_train, y_train)), 10))
print('RMSE Test = ', round(np.sqrt(mean_squared_error(y_predicted, y_test)), 10))
print('Correlation In-Sample Predicted/Train = ', round(np.corrcoef(np.reshape(y_predicted_train, (-1)), y_train)[0][1], 3))
print('Correlation Out-of-Sample Predicted/Test = ', round(np.corrcoef(np.reshape(y_predicted, (-1)), np.reshape(y_test, (-1)))[0][1], 3))
print('Model Bias = ', round(model_bias(y_predicted), 2))
print('---')
