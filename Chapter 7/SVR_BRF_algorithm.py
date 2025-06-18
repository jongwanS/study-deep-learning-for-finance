import numpy as np
import matplotlib.pyplot as plt

class RBF_SVR:
    def __init__(self, C=1.0, epsilon=0.5, gamma=0.5, lr=0.01, epochs=100):
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.epochs = epochs

    def rbf_kernel(self, x1, x2):
        return np.exp(-self.gamma * (x1 - x2) ** 2)

    def fit(self, X, y):
        self.X = np.array(X).flatten()
        self.y = np.array(y)
        n = len(self.X)
        self.alpha = np.zeros(n)
        self.alpha_star = np.zeros(n)
        self.b = 0.0

        for epoch in range(self.epochs):
            for i in range(n):
                xi = self.X[i]
                yi = self.y[i]

                # 예측값 계산
                y_pred = sum(
                    (self.alpha[j] - self.alpha_star[j]) * self.rbf_kernel(self.X[j], xi)
                    for j in range(n)
                ) + self.b

                error = y_pred - yi

                # 오차가 epsilon을 넘으면 alpha 조정
                if error > self.epsilon:
                    self.alpha[i] = min(self.C, self.alpha[i] + self.lr)
                elif error < -self.epsilon:
                    self.alpha_star[i] = min(self.C, self.alpha_star[i] + self.lr)

                # 간단한 b 조정
                self.b -= self.lr * error * 0.01

    def predict(self, X):
        X = np.array(X).flatten()
        y_pred = []
        for x in X:
            result = sum(
                (self.alpha[i] - self.alpha_star[i]) * self.rbf_kernel(self.X[i], x)
                for i in range(len(self.X))
            ) + self.b
            y_pred.append(result)
        return np.array(y_pred)

# ▶ 데이터셋
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.5, 4.8, 6.3, 8.2, 10.0])

# ▶ 모델 학습 및 예측
model = RBF_SVR(C=1.0, epsilon=0.5, gamma=0.5, lr=0.05, epochs=200)
model.fit(X, y)
y_pred = model.predict(X)

# ▶ 시각화
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Custom SVR (RBF)')
plt.title("Custom RBF Kernel SVR")
plt.legend()
plt.grid(True)
plt.show()
