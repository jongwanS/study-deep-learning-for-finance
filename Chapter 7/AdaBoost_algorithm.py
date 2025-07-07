import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 결정 stump (단일 분할 회귀 트리)
class DecisionStumpRegressor:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        min_error = float('inf')

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_value = np.average(y[left_mask], weights=sample_weights[left_mask])
                right_value = np.average(y[right_mask], weights=sample_weights[right_mask])

                predictions = np.where(left_mask, left_value, right_value)
                error = np.sum(sample_weights * (y - predictions) ** 2)

                if error < min_error:
                    min_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.left_value = left_value
                    self.right_value = right_value

    def predict(self, X):
        feature = X[:, self.feature_index]
        return np.where(feature <= self.threshold, self.left_value, self.right_value)


# AdaBoost 회귀 모델
class AdaBoostRegressor:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = len(y)
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            stump = DecisionStumpRegressor()
            stump.fit(X, y, weights)
            predictions = stump.predict(X)

            # 상대 오차 계산
            error = np.abs(y - predictions)
            error_max = np.max(error)
            if error_max == 0:
                break
            error /= error_max

            # 전체 가중 오차
            weighted_error = np.sum(weights * error)

            # 모델 기여도 (alpha) 계산
            if weighted_error >= 0.5:
                break  # 성능이 너무 낮음 (중단)
            alpha = 0.5 * np.log((1 - weighted_error) / (weighted_error + 1e-10))

            # 가중치 업데이트
            weights *= np.exp(alpha * error)
            weights /= np.sum(weights)

            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        final_pred = np.zeros(X.shape[0])
        for model, alpha in zip(self.models, self.alphas):
            final_pred += alpha * model.predict(X)
        return final_pred / np.sum(self.alphas)


# -------------------------
# 테스트 코드
# -------------------------

# 1. 데이터 생성 (회귀용 샘플 데이터)
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)

# 2. 모델 학습
model = AdaBoostRegressor(n_estimators=20)
model.fit(X, y)

# 3. 예측
y_pred = model.predict(X)

# 4. 성능 평가
print("Mean Squared Error:", mean_squared_error(y, y_pred))

# 5. 결과 시각화
plt.scatter(X, y, color='blue', label='Actual')
plt.scatter(X, y_pred, color='red', label='Predicted')
plt.title("AdaBoost Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
