import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyRegressor
from master_function import data_preprocessing
from master_function import plot_train_test_values, calculate_accuracy, model_bias
from sklearn.metrics import mean_squared_error

# importing the time series
# Import the data (write the code in one line)
data = np.array(pd.read_excel('Daily_EURUSD_Historical_Data.xlsx')['<CLOSE>'])
# Difference the data and make it stationary
'''
#### diff 함수
diff[0] = data[1] - data[0]
diff[1] = data[2] - data[1]
diff[2] = data[3] - data[2]
'''
data = np.diff(data)

# Setting the hyperparameters
'''
하이퍼파라미터(hyperparameter) 는 모델 학습 전에 직접 설정해야 하는 값
> 모델이 어떻게 학습할지 조절하는 외부 변수
'''
num_lags = 500 # 과거 500개 데이터를 입력으로 사용해서 예측하겠다는 의미
train_test_split = 0.80 #전체 데이터 중 80%는 학습(training)에, 나머지 20%는 테스트(test)에 사용

# Creating the training and test sets
x_train, y_train, x_test, y_test = data_preprocessing(data, num_lags, train_test_split)

# Fitting the model
model = DummyRegressor(strategy = 'mean')
model.fit(x_train, y_train)

# Predicting in-sample
'''
-1는, 열은 1개로 고정
'''
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