import matplotlib.pyplot as plt
import numpy as np

# 가상의 손실 함수 (예: y = x^2)
x = np.linspace(-3, 3, 100)
y = x**2

# 현재 파라미터 위치
current_x = -2
current_y = current_x**2

# 기울기 (미분: dy/dx = 2x)
gradient = 2 * current_x

# 학습률 (learning rate)
lr = 0.5

# 다음 위치 계산
next_x = current_x - lr * gradient
next_y = next_x**2

# 시각화
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Loss Function: $y = x^2$')
plt.scatter(current_x, current_y, color='red', label='Current Position', zorder=5)
plt.scatter(next_x, next_y, color='green', label='Next Position', zorder=5)
plt.arrow(current_x, current_y,
          next_x - current_x, next_y - current_y,
          head_width=0.2, head_length=0.3, fc='blue', ec='blue',
          label='Gradient Descent Step')

plt.title('Gradient Descent Visualization on a Loss Function')
plt.xlabel('Parameter (x)')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# 2D 손실 함수 정의: z = x^2 + y^2
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

# 시작점
start_x, start_y = -2.5, -2.0
learning_rate = 0.2
steps = 10
points = [(start_x, start_y)]

# 기울기 하강법
x_curr, y_curr = start_x, start_y
for _ in range(steps):
    grad_x = 2 * x_curr
    grad_y = 2 * y_curr
    x_curr -= learning_rate * grad_x
    y_curr -= learning_rate * grad_y
    points.append((x_curr, y_curr))

# 시각화
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Loss (z)')

# 이동 경로 시각화
px, py = zip(*points)
pz = [x**2 + y**2 for x, y in points]
ax.plot(px, py, pz, color='red', marker='o', label='Gradient Descent Path')

plt.legend()
plt.title('2D Loss Surface: z = x² + y²')
plt.tight_layout()
plt.show()
