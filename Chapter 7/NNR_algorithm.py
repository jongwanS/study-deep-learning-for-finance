import numpy as np
import matplotlib.pyplot as plt


class KNearestNeighborsRegressor:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.ravel(X) # 1차원으로 바꿔줌
        self.y_train = np.array(y)

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = []

        for x in X_test:
            # 거리 계산 (유클리디안 거리)
            # 유클리디안 거리 = 절대값
            distances = np.abs(self.X_train - x)

            # 가까운 K개 인덱스
            #argsort()[:k]
            k_indices = distances.argsort()[:self.k]

            # K개의 타깃 평균
            k_avg = np.mean(self.y_train[k_indices])
            predictions.append(k_avg)

        return np.array(predictions)


# 샘플 데이터
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.5, 4.8, 6.3, 8.2, 10.0])

# 예측을 위한 범위 (시각화를 위해 촘촘한 X축)
"""
np.linspace(0, 6, 200)
> 0부터 6까지 균등하게 200개의 숫자
> 예를 들어, 0, 0.03, 0.06, ..., 5.97, 6 이런 식으로 200개의 숫자가 균등 분포된 1차원 배열
.reshape(-1, 1)은 행 크기는 자동 계산(-1), 열 크기는 1로 바꾸겠다는 뜻
"""
X_test = np.linspace(0, 6, 200).reshape(-1, 1)

# 모델 생성 및 학습
model = KNearestNeighborsRegressor(k=2)
model.fit(X, y)
y_pred = model.predict(X_test)

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_test, y_pred, color='red', label='KNN Prediction (k=2)')
plt.title("K-Nearest Neighbors Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
