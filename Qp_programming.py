import cvxpy as cp

# 변수 정의
x = cp.Variable()
y = cp.Variable()

'''

'''
# 목적 함수 (이차식)
objective = cp.Minimize(x**2 + y**2)

# 제약 조건
constraints = [
    x + y >= 4,
    x >= 0,
    y >= 0
]

# 문제 정의 및 풀기
prob = cp.Problem(objective, constraints)
prob.solve()

# 결과 출력
print("최적의 x:", x.value)
print("최적의 y:", y.value)
print("최소값:", prob.value)
