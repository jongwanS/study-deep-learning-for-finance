import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# -----------------------------
# 사용자 정의 함수 정의 또는 임포트
# -----------------------------

def data_preprocessing(data, num_lags=10, train_test_split=0.8):
    X, y = [], []
    for i in range(num_lags, len(data)):
        X.append(data[i - num_lags:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    split = int(len(X) * train_test_split)
    return X[:split], y[:split], X[split:], y[split:]

def calculate_accuracy(y_pred, y_true):
    return (1 - np.mean(np.abs((y_true - y_pred) / y_true))) * 100

def model_bias(pred):
    return np.mean(pred)

def plot_train_test_values(n_train, n_test, y_train, y_test, y_pred_test):
    plt.figure(figsize=(14, 5))
    plt.plot(range(n_train), y_train[-n_train:], label='Train')
    plt.plot(range(n_train, n_train + n_test), y_test[:n_test], label='Test')
    plt.plot(range(n_train, n_train + n_test), y_pred_test[:n_test], label='Prediction')
    plt.legend()
    plt.title("Gradient Boosted Tree Time Series Forecast")
    plt.show()

# -----------------------------
# 간단한 Gradient Boosting Regressor 구현 (XGB 대체)
# -----------------------------
class SimpleGBR:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        from sklearn.tree import DecisionTreeRegressor

        pred = np.zeros_like(y, dtype=float)

        for i in range(self.n_estimators):
            residual = y - pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)
            update = tree.predict(X)
            pred += self.learning_rate * update
            self.models.append(tree)

    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for tree in self.models:
            pred += self.learning_rate * tree.predict(X)
        return pred

# -----------------------------
# 데이터 불러오기 및 전처리
# -----------------------------
np.random.seed(42)
data = np.cumsum(np.random.randn(2000))

num_lags = 50
split_ratio = 0.8
x_train, y_train, x_test, y_test = data_preprocessing(data, num_lags, split_ratio)

# -----------------------------
# 간단한 Gradient Boosting 회귀 모델 훈련
# -----------------------------
model = SimpleGBR(n_estimators=50, learning_rate=0.1, max_depth=3)
model.fit(x_train, y_train)

# -----------------------------
# 예측
# -----------------------------
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

# -----------------------------
# 평가 지표 출력
# -----------------------------
print('---')
print('Accuracy Train =', round(calculate_accuracy(y_pred_train, y_train), 2), '%')
print('Accuracy Test =', round(calculate_accuracy(y_pred_test, y_test), 2), '%')
print('RMSE Train =', round(np.sqrt(mean_squared_error(y_pred_train, y_train)), 6))
print('RMSE Test =', round(np.sqrt(mean_squared_error(y_pred_test, y_test)), 6))
print('Correlation In-Sample =', round(np.corrcoef(y_pred_train, y_train)[0][1], 3))
print('Correlation Out-of-Sample =', round(np.corrcoef(y_pred_test, y_test)[0][1], 3))
print('Model Bias =', round(model_bias(y_pred_test), 2))
print('---')

# -----------------------------
# 시각화
# -----------------------------
plot_train_test_values(100, 50, y_train, y_test, y_pred_test)