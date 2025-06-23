import numpy as np
import matplotlib.pyplot as plt

# 간단한 결정 트리 회귀 모델
class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # 랜덤으로 선택할 feature 수
        self.tree = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_input(x, self.tree) for x in X])

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _best_split(self, X, y):
        best_mse = float('inf')
        best_split = None
        n_samples, n_features = X.shape

        # 🎯 무작위로 일부 특성 선택
        n_selected = self.max_features or int(np.sqrt(self.n_features)) or 1
        feature_indices = np.random.choice(self.n_features, n_selected, replace=False)

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                if sum(left_idx) == 0 or sum(right_idx) == 0:
                    continue
                left_mse = self._mse(y[left_idx])
                right_mse = self._mse(y[right_idx])
                mse = (left_mse * sum(left_idx) + right_mse * sum(right_idx)) / n_samples

                if mse < best_mse:
                    best_mse = mse
                    best_split = (feature, threshold)

        return best_split

    def _build_tree(self, X, y, depth):
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            len(set(y)) == 1):
            return np.mean(y)

        split = self._best_split(X, y)
        if not split:
            return np.mean(y)

        feature, threshold = split
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return {"feature": feature, "threshold": threshold, "left": left, "right": right}

    def _predict_input(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_input(x, tree["left"])
        else:
            return self._predict_input(x, tree["right"])

# 트리 구조를 텍스트 형태로 출력하는 함수
def print_tree(tree, depth=0):
    indent = "  " * depth
    if not isinstance(tree, dict):
        print(f"{indent}Leaf: {tree:.3f}")
        return
    print(f"{indent}Feature[{tree['feature']}] <= {tree['threshold']:.3f}?")
    print(f"{indent}--> True:")
    print_tree(tree['left'], depth + 1)
    print(f"{indent}--> False:")
    print_tree(tree['right'], depth + 1)


# 랜덤 포레스트 회귀 모델
class RandomForestRegressor:
    def __init__(self, n_estimators=10, max_depth=3, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)


# ✅ 테스트 예제
if __name__ == "__main__":
    # 간단한 데이터셋
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([1.2, 1.9, 3.0, 4.1, 5.1, 6.2])

    # 모델 학습
    model = RandomForestRegressor(n_estimators=3, max_depth=3)
    model.fit(X, y)

    # 예측
    preds = model.predict(X)
    print("예측값:", preds)

    # 첫 번째 트리 구조 출력
    print("\n=== 첫 번째 트리 구조 ===")
    print_tree(model.trees[0].tree)
