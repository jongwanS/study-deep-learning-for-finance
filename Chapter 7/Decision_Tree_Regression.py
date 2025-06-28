import numpy as np
import matplotlib.pyplot as plt

class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    # MSE 계산
    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    # 최적의 분할 찾기
    def _best_split(self, X, y):
        best_mse = float('inf')
        best_idx = None
        best_val = None

        for feature_index in range(X.shape[1]): #X.shape[1] 열(column)의 개수
            values = np.unique(X[:, feature_index]) ## 0번째 열의 유일값 print(np.unique(X[:, 0]))
            for val in values:
                left_idx = X[:, feature_index] <= val
                right_idx = X[:, feature_index] > val

                y_left, y_right = y[left_idx], y[right_idx]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                mse = (len(y_left) * self._mse(y_left) + len(y_right) * self._mse(y_right)) / len(y)
                if mse < best_mse:
                    best_mse = mse
                    best_idx = feature_index
                    best_val = val

        return best_idx, best_val

    # 트리 재귀 구성
    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return np.mean(y)

        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            return np.mean(y)

        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold

        left_branch = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_branch = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': left_branch,
            'right': right_branch
        }

    def fit(self, X, y):
        self.tree = self._build_tree(np.array(X), np.array(y), 0)

    def _predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


# 간단한 1차원 데이터
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([1.1, 1.9, 3.0, 4.1, 5.0, 6.0, 7.2, 8.1, 9.0, 10.1])

# 모델 생성 및 학습
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

# 예측
X_test = np.linspace(1, 10, 100).reshape(-1, 1)
y_pred = tree.predict(X_test)

# 시각화
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X_test, y_pred, color='red', label='Prediction')
plt.title("Decision Tree Regressor (Custom)")
plt.legend()
plt.show()