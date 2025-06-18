import numpy as np
import matplotlib.pyplot as plt


class SimpleSVR:
    """
    하이퍼파라미터
    - lr: 학습률 (learning rate)
     > 학습률은 모델 파라미터 𝑤,𝑏 w,b 를 얼마나 크게, 빠르게 업데이트할지 결정하는 숫자
     > 너무 크면? 한 번에 너무 크게 움직여서 최적값을 넘어서거나 불안정하게 학습
     > 너무 작으면? 업데이트가 너무 조금씩 되어서 학습 속도가 느려지고, 최적값에 도달하는 데 시간이 오래 걸림
    - epochs: 전체 학습 반복 수
    - epsilon: ε-튜브 범위 (무시할 수 있는 오차)
    - C: 정규화 강도 (과적합 방지)
     > C는 SVR 모델에서 과적합(overfitting)을 얼마나 막을지, 오차에 얼마나 민감할지를 조절
     > 너무 꼼꼼하면 훈련 데이터에 너무 맞춰서 새로운 데이터에 약해짐
     > 너무 대충하면 훈련 데이터도 잘 못 맞춤.
    - w, b: 모델 파라미터 (가중치와 절편)
    """
    def __init__(self, lr=0.01, epochs=100, epsilon=0.5, C=1.0):
        self.lr = lr
        self.epochs = epochs
        self.epsilon = epsilon
        self.C = C
        self.w = 0.0
        self.b = 0.0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n = len(X)

        for epoch in range(self.epochs):
            for i in range(n):
                """
                y_pred와 error를 계산
                """
                x_i = X[i]
                y_i = y[i]
                y_pred = self.w * x_i + self.b
                error = y_pred - y_i

                """
                ε-튜브 안에 있는 오차는 무시.
                바깥에 있는 경우만 가중치를 업데이트하기 위한 gradient를 계산.
                
                error > ε: 예측이 실제보다 너무 크면, 가중치와 절편을 감소시키는 방향으로 업데이트
                error < -ε: 예측이 실제보다 너무 작으면, 가중치와 절편을 증가시키는 방향으로 업데이트
                -ε ≤ error ≤ ε: 오차가 허용범위 안이라 업데이트하지 않음 (gradient = 0)
                """
                if error > self.epsilon: # 튜브 밖 윗부분
                    grad_w = x_i
                    grad_b = 1
                elif error < -self.epsilon: # 튜브 밖 아랫 부분
                    grad_w = -x_i
                    grad_b = -1
                else:
                    grad_w = 0
                    grad_b = 0

                # 경사하강법을 통해 w, b를 업데이트
                self.w -= self.lr * self.C * grad_w
                self.b -= self.lr * self.C * grad_b

    def predict(self, X):
        return self.w * np.array(X) + self.b


# ▶ 샘플 데이터
X = np.array([1, 2, 3, 4, 5])
y = np.array([2.5, 4.8, 6.3, 8.2, 10.0])

# ▶ 모델 학습
model = SimpleSVR(lr=0.01, epochs=100, epsilon=0.5, C=1.0)
model.fit(X, y)
y_pred = model.predict(X)

# ▶ 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='SVR Prediction')
plt.plot(X, y_pred + model.epsilon, 'k--', label='Epsilon-tube')
plt.plot(X, y_pred - model.epsilon, 'k--')

plt.title("Simple SVR with Epsilon-Insensitive Loss")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
