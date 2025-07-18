import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# 예시 입력, 가중치 초기화
X = np.array([0.5, 0.1])     # 입력층 2개
y = np.array([1])            # 타깃

# 가중치 (입력층->은닉층 3개)
'''
W1[0, 0] = 0.4: X1에서 H1으로 가는 연결의 가중치
W1[0, 1] = 0.3: X2에서 H1으로 가는 연결의 가중치
W1[1, 0] = 0.2: X1에서 H2로 가는 연결의 가중치
W1[1, 1] = 0.1: X2에서 H2로 가는 연결의 가중치
W1[2, 0] = 0.5: X1에서 H3으로 가는 연결의 가중치
W1[2, 1] = -0.4: X2에서 H3으로 가는 연결의 가중치
'''
W1 = np.array([
    [0.4, 0.3],
    [0.2, 0.1],
    [0.5, -0.4]
])  # (3,2)
b1 = np.array([0.1, 0.2, -0.1])  # (3,)

# 가중치 (은닉층->출력층 1개)
W2 = np.array([[0.7, 0.6, -0.5]])  # (1,3)
b2 = np.array([0.3])                # (1,)

# 순전파
Z1 = W1.dot(X) + b1   # (3,)
A1 = sigmoid(Z1)      # (3,)

Z2 = W2.dot(A1) + b2  # (1,)
A2 = sigmoid(Z2)      # (1,)

# 역전파
dL_dA2 = A2 - y
dZ2 = dL_dA2 * sigmoid_derivative(Z2)

dL_dW2 = dZ2.reshape(-1,1) * A1.reshape(1,-1)
dL_db2 = dZ2

dL_dA1 = W2.T * dZ2
dZ1 = dL_dA1 * sigmoid_derivative(Z1)

dL_dW1 = dZ1.reshape(-1,1) * X.reshape(1,-1)
dL_db1 = dZ1

# 시각화 함수
def draw_network_forward_backward(X, A1, A2, dZ1, dZ2):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()

    pos = {
        'X1': (0, 2),
        'X2': (0, 1),
        'H1': (1, 3),
        'H2': (1, 2),
        'H3': (1, 1),
        'Y':  (2, 2),
    }

    for node in pos:
        G.add_node(node)

    forward_edges = [
        ('X1', 'H1'), ('X1', 'H2'), ('X1', 'H3'),
        ('X2', 'H1'), ('X2', 'H2'), ('X2', 'H3'),
        ('H1', 'Y'), ('H2', 'Y'), ('H3', 'Y')
    ]

    backward_edges = [(t,s) for s,t in forward_edges]

    plt.figure(figsize=(10,6))
    ax = plt.gca()
    ax.set_title('Visualizing Forward and Backward Propagation in a 2-Layer Neural Network', fontsize=14)

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightgray')

    # 역전파 후 dZ1, dZ2 평탄화
    dZ1 = dZ1.flatten()
    dZ2 = dZ2.flatten()

    labels = {
        'X1': f'X1\n{X[0]:.2f}',
        'X2': f'X2\n{X[1]:.2f}',
        'H1': f'H1\n{A1[0]:.3f}\ngrad\n{dZ1[0]:.3f}',
        'H2': f'H2\n{A1[1]:.3f}\ngrad\n{dZ1[1]:.3f}',
        'H3': f'H3\n{A1[2]:.3f}\ngrad\n{dZ1[2]:.3f}',
        'Y': f'Y\n{A2[0]:.3f}\ngrad\n{dZ2[0]:.3f}',
    }
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    # =========================================================
    # 순전파 선 (파란색)
    nx.draw_networkx_edges(G, pos, edgelist=forward_edges,
                           edge_color='blue', arrowsize=20, arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.08', # <-- 이 부분을 추가/수정: 양수 값으로 약간 휘게
                           alpha=0.8) # 투명도 조절 (옵션)

    # 역전파 선 (빨간색)
    nx.draw_networkx_edges(G, pos, edgelist=backward_edges,
                           edge_color='red', style='dashed', arrowsize=20, arrowstyle='-|>',
                           connectionstyle='arc3,rad=-0.03', # <-- 이 부분을 추가/수정: 음수 값으로 반대 방향으로 휘게
                           alpha=0.8) # 투명도 조절 (옵션)
    # =========================================================

    forward_patch = mpatches.Patch(color='blue', label='forward propagation')
    backward_patch = mpatches.Patch(color='red', label='back propagation')
    plt.legend(handles=[forward_patch, backward_patch], loc='upper right')

    plt.axis('off')
    plt.show()

# 함수 호출
draw_network_forward_backward(X, A1, A2, dZ1, dZ2)