import numpy as np
'''
| 의미       | 파이썬 변수 | 값                | 설명                            |
| -------- | ------ | ---------------- | ----------------------------- |
| 목적 함수 계수 | `c`    | `[40, 30]`       | `Z = 40x + 30y`               |
| 제약 조건 계수 | `A`    | `[[2,1], [1,2]]` | `2x + y ≤ 100`, `x + 2y ≤ 80` |
| 제약 조건 우변 | `b`    | `[100, 80]`      | 각각의 ≤ 오른쪽 값                   |
'''
def simplex(c, A, b):
    num_vars = len(c)
    num_constraints = len(b)

    # Slack 변수 추가
    '''
    선형계획 문제에서 부등식 제약 조건(<=)을 등식(=)으로 바꾸기 위해 여유변수(slack variable) 를 추가
    2x + y ≤ 100   →   2x + y + s1 = 100
    x + 2y ≤ 80    →   x + 2y + s2 = 80
    '''
    A = np.hstack([A, np.eye(num_constraints)])
    c = np.concatenate([c, np.zeros(num_constraints)])

    # 초기 테이블 생성
    tableau = np.zeros((num_constraints + 1, len(c) + 1))
    tableau[:-1, :-1] = A
    tableau[:-1, -1] = b
    tableau[-1, :-1] = -c

    while True:
        # 1. 들어올 변수 (가장 작은 음수 계수)
        col = np.argmin(tableau[-1, :-1])
        if tableau[-1, col] >= 0:
            break  # 최적 도달

        # 2. 나갈 변수 (최소 ratio test)
        ratios = []
        for i in range(num_constraints):
            if tableau[i, col] > 0:
                ratios.append(tableau[i, -1] / tableau[i, col])
            else:
                ratios.append(np.inf)
        row = np.argmin(ratios)
        if ratios[row] == np.inf:
            raise Exception("해가 없음 (Unbounded)")

        # 3. Pivot (가우스-조르당 방식)
        pivot = tableau[row, col]
        tableau[row, :] /= pivot
        for i in range(len(tableau)):
            if i != row:
                tableau[i, :] -= tableau[i, col] * tableau[row, :]

    # 결과 추출
    result = np.zeros(len(c))
    for i in range(num_constraints):
        col_idx = np.where(tableau[i, :-1] == 1)[0]
        if len(col_idx) == 1 and all(tableau[:, col_idx[0]][np.arange(len(tableau)) != i] == 0):
            result[col_idx[0]] = tableau[i, -1]

    optimal_value = tableau[-1, -1]
    return result[:num_vars], optimal_value

# 예제 문제 설정
'''
목적 함수의 계수
- Z=40x+30y
'''
c = np.array([40, 30])  # 목적 함수 계수
'''
제약조건 계수
- 2x + y ≤ 100
- x + 2y ≤ 80
'''
A = np.array([
    [2, 1],
    [1, 2]
])
#제약 조건
b = np.array([100, 80])

# 실행
solution, optimal = simplex(c, A, b)
print(f"최적 해: x = {solution[0]:.2f}, y = {solution[1]:.2f}")
print(f"최대 이익: {optimal:.2f} 만원")
