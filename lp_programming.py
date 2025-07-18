import numpy as np
'''
| 의미       | 파이썬 변수 | 값                | 설명                            |
| -------- | ------ | ---------------- | ----------------------------- |
| 목적 함수 계수 | `c`    | `[40, 30]`       | `Z = 40x + 30y`               |
| 제약 조건 계수 | `A`    | `[[2,1], [1,2]]` | `2x + y ≤ 100`, `x + 2y ≤ 80` |
| 제약 조건 우변 | `b`    | `[100, 80]`      | 각각의 ≤ 오른쪽 값                   |
'''
def simplex(c, A, b):
    num_vars = len(c) # 2
    num_constraints = len(b) #

    # Slack 변수 추가
    '''
    선형계획 문제에서 부등식 제약 조건(<=)을 등식(=)으로 바꾸기 위해 여유변수(slack variable) 를 추가
    2x + y ≤ 100   →   2x + y + s1 = 100
    x + 2y ≤ 80    →   x + 2y + s2 = 80
    '''
    # 단위행렬 생성
    '''
    np.eye 단위 행렬 (identity matrix) 을 생성하는 함수
    단위행렬 :  대각선에 1, 나머지 요소는 0인 정사각 행렬
    단위행렬 사용이유 : 슬랙 변수의 계수를 단위 행렬 형태로 추가하기 위해 사용
    
    np.hstack()은 수평(horizontal)으로 배열을 이어붙이는 함수
    A : [[2, 1, 1, 0],
         [1, 2, 0, 1]]
    '''
    A = np.hstack([A, np.eye(num_constraints)])

    '''
    np.zeros({number} => number 갯수만큼 [0,0 ....]
    > 슬랙 변수 s1, s2는 목적 함수에 영향을 주지 않으므로 계수는 0이야!
    np.concatenate([...])	전체 목적 함수 벡터 완성 ([40, 30, 0, 0])
    c결과 : [40, 30, 0, 0]
    '''
    c = np.concatenate([c, np.zeros(num_constraints)])

    # 초기 테이블 생성
    tableau = np.zeros((num_constraints + 1, len(c) + 1)) #3x5 생성
    tableau[:-1, :-1] = A #A를 테이블의 제약 조건 위치에 복사, :-1, :-1 마지막 행&열 제외
    tableau[:-1, -1] = b # tableau[:-1, -1]: 마지막 열 (RHS 열, = 등호의 뜻)에 복사
    tableau[-1, :-1] = -c # tableau[-1, :-1]: 마지막 행에 목적함수 넣기
    '''
    A : [[2, 1, 1, 0], [1, 2, 0, 1]]
    b : [[100, 80]]
    c : [40, 30, 0, 0]
    tableau =   [[  2.   1.   1.   0. 100.]
                 [  1.   2.   0.   1.  80.]
                 [-40. -30.   0.   0.   0.]]
    '''

    while True:
        # 1. 들어올 변수 (가장 작은 음수 계수)
        # tableau[-1, :-1] -> tableau[-1] : 마지막행, :-1 RHS 제외 ==> [-40. -30. 0. 0.]
        # np.argmin : 최소값의 인덱스
        col = np.argmin(tableau[-1, :-1])
        if tableau[-1, col] >= 0:
            break  # 최적 도달

        # 2. 나갈 변수 (최소 ratio test)
        ratios = []
        for i in range(num_constraints):
            if tableau[i, col] > 0:
                ratios.append(tableau[i, -1] / tableau[i, col]) #tableau[i, -1] 는 i번째 행의 "맨 마지막 열
            else:
                ratios.append(np.inf)
        row = np.argmin(ratios)# 최소값의 배열 인덱스를 가져옴
        if ratios[row] == np.inf: # np.inf : 엄청나게 큰 값
            raise Exception("해가 없음 (Unbounded)")

        # 3. Pivot (가우스-조르당 방식)
        pivot = tableau[row, col]
        tableau[row, :] /= pivot
        for i in range(len(tableau)): #len(tableau) : 3
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
