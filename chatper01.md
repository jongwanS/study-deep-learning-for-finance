#### github_example : https://github.com/sofienkaabar/deep-learning-for-finance

## Chapter 1. Introducing Data Science and Trading
### Understanding Data
- Data
  - `simplest and purest form`
  - a collection of raw information that can be of any type
- The final aim
  - `collecting data is decision-making`

- 표1)  
  - 평균 수익률 : 5.16
  
| Stock | Dividend Yield |
|-------|----------------|
| A     | 5.20%          |
| C     | 4.12%          |
| D     | 6.94%          | 
| E     | 5.55%          |

- 표2)  
  - 내년 positive 확률 : 88%  
  - 내년 negative 확률 : 12%  
  
| Year | Correlation |
|------|-------------|
| 2015 | Positive    |
| 2016 | Positive    |
| 2017 | Positive    |
| 2018 | Negative    |
| 2019 | Positive    |
| 2020 | Positive    |
| 2021 | Positive    |
| 2022 | Positive    |
| 2023 | Positive    |

- what types of data can be used and segment them into **different groups**
  - `Numerical data`
  - `Categorical data`
    - ex) blood type, eye color
  - `Text data`
  - `Visual data`
  - `Audio data`

- Data science is a `transdisciplinary field` that tries to extract intelligence and conclusions from data using different techniques and models

- The data science process is composed of `many steps` besides just analyzing data
  - `Data gathering`
    - 목표: 신뢰할 수 있고 정확한 출처에서 데이터를 얻는 것
    - 중요성: **Garbage in, garbage out**라는 표현처럼, 만약 부정확하거나 잘못된 데이터를 수집하면 이후 모든 분석과 결과가 무효가 될 수 있습니다.
  - `Data preprocessing`
    - 목표: 데이터를 모델이 사용할 수 있는 형식으로 정리하고 준비하는 것
    - 작업 예시: 불필요한 데이터 제거, 결측값 추가, 잘못되거나 중복된 데이터 제거, 정규화 및 잡음 제거 등의 복잡한 작업이 포함됩니다.
    - 중요성: 정확하고 일관된 데이터를 준비하는 것이 중요한 이유는 모델이 잘 작동할 수 있게 하기 위해서입니다.
  - `Data exploration`
    - 목표: 데이터를 이해하고, 그 안에 숨어 있는 패턴이나 트렌드를 찾기 위해 기초적인 통계 분석을 수행하는 단계
    - 작업 예시: 데이터의 평균(mean), 중앙값(median), 분산(variance) 등의 기초적인 통계치를 계산하여 데이터의 특성을 파악합니다.
    - 중요성: 데이터의 분포나 특징을 이해하고, 분석에 중요한 지표를 식별하는 데 필요합니다.
  - `Data visualization`
    - 목표: 데이터를 시각적으로 표현하여 패턴, 경향, 이상치를 쉽게 확인하고 해석할 수 있도록 돕는 것
    - 작업 예시: 히스토그램, 히트맵, 상자 그림(Box plot) 등을 활용해 데이터를 시각적으로 표현합니다.
    - 중요성: 데이터의 트렌드를 한눈에 볼 수 있어 분석 결과를 더 명확하게 이해하고 해석할 수 있습니다.
  - `Data analysis`
    - 목표: 데이터를 다양한 학습 모델에 적용하여 미래의 결과를 예측하거나 분류하는 것
    - 작업 예시: 데이터 모델링, 학습 알고리즘 적용(예: 회귀, 분류, 군집화 등)
    - 중요성: 이 단계에서 모델을 학습시키고, 데이터를 분석하여 예측하거나 중요한 결론을 도출합니다.
  - `Data interpretation`
    - 목표: 모델이 제공하는 결과를 이해하고, 얻은 인사이트를 실용적으로 해석하는 단계
    - 작업 예시: 결과를 평가하고, 최적화가 필요하면 모델을 다시 실행하거나 파라미터를 조정합니다.
    - 중요성: 분석 결과를 바탕으로 실용적인 결정을 내리고, 반복적인 과정이 필요할 수 있습니다. 최적화를 통해 결과를 개선하고 재평가합니다.

### Understanding Data Science
- Data science plays an essential role in technology and progress
  - 알고리즘은 작업을 수행하기 위해 데이터 과학 도구로부터 제공된 정보를 기반으로 작동
    - 알고리즘이란, 특정 작업을 수행하거나 특정 문제를 해결하기 위해 설계된 순서 있는 절차들의 집합
- 알고리즘 `예시`
  - 필요한 금융 데이터를 차트 플랫폼에 업데이트
    ````text
    서버와 온라인 데이터 제공자를 연결한다.  
    가장 최근 타임스탬프가 있는 금융 데이터를 복사한다.  
    해당 데이터를 차트 플랫폼에 붙여넣는다.  
    다시 1단계로 돌아가 이 과정을 반복한다.
    ````
  - 차익거래 알고리즘 
    ````text
    거래소 A의 주식 가격: $10.00
    거래소 B의 주식 가격: $10.50
    이 경우 차익거래 알고리즘은 다음과 같이 작동합니다:
    거래소 A에서 $10.00에 주식을 구매
    거래소 B에서 즉시 $10.50에 판매
    $0.50의 차익을 확보하고 이 과정을 반복하여 가격 차가 사라질 때까지 계속
    ````


 > **알고리즘의 본질** : 유한하거나 무한한 목표를 위해 특정 지시사항을 수행

- `데이터 과학의 두 가지 주요 용도`
  - 데이터 해석 (Data Interpretation)
    - 알고리즘을 사용하는 목적은 데이터를 `무엇`과 `어떻게`의 관점에서 **이해**하기 위함
  - 데이터 예측 (Data Prediction)
    - 알고리즘을 사용하는 목적은 데이터의 **다음에 무슨 일이 일어날지를 파악**하기 위함

- 이 책에서 다룰 알고리즘
  - `지도 학습 (Supervised Learning)`
    - 라벨이 있는 데이터가 필요
    - 모델이 **과거 데이터를 학습**해 패턴을 이해하고, **새로운 데이터에 대해 예측**을 수행
    - 예: 선형 회귀(Linear Regression), 랜덤 포레스트(Random Forest)
  - `비지도 학습 (Unsupervised Learning)`
    - 라벨 없는 데이터로 작동
    - 모델이 스스로 숨겨진 패턴을 찾음
    - 예: 군집화(Clustering), 주성분 분석(PCA)
  - `강화 학습 (Reinforcement Learning)`
    - 데이터를 필요로 하지 않음.
    - 환경과 상호작용하며 보상 시스템을 통해 스스로 학습
    - 일반적으로 시간 시계열 예측보다는 에이전트가 행동을 최적화하는 정책 개발에 사용

- 데이터 과학 알고리즘은 금융 외에도 다양한 분야에서 활용
  - 비즈니스 분석: 가격 최적화, 고객 이탈 예측, 마케팅 성과 향상
  - 헬스케어: 환자 치료 개선, 신약 개발, 비용 절감
  - 스포츠: 팀 성과 분석, 선수 스카우트, 경기 예측
  - 연구: 과학적 발견 지원, 이론 검증, 지식 획득

- **데이터 과학자**는 복잡한 데이터를 분석하고 인사이트를 얻어, 의사결정에 도움이 되는 정보를 제공
  - 통계 모델 개발
  - 머신러닝 기법 적용
  - 데이터 시각화
  - 데이터 기반 솔루션 구현 지원 및 결과 전달

### Introduction to Financial Markets and Trading
> Note : 데이터 과학자와 데이터 엔지니어의 차이점
> 데이터 과학자는 데이터 분석과 해석에 초점을 맞춤
> 데이터 엔지니어는 데이터 수집, 저장, 분석을 위한 인프라 구축에 집중

- 금융 상품은 현물(spot) 또는 파생상품(derivatives) 형태로 존재
  - 선도계약/선물(futures): 특정 자산을 미래에 지정된 가격에 매수하기로 약정하는 계약
  - 옵션(options): 특정 자산을 미래에 정해진 가격으로 매수할 권리(의무는 없음)를 가지며, 현재 프리미엄(옵션 비용)을 지불

#### 주요 자산군(Asset Classes)
- 주식(Stock markets)
  - 기업이 자금을 조달하기 위해 발행하는 주식을 거래하는 시장.
  - 주식을 보유하면 기업의 일부를 소유하게 되며, 배당이나 의결권을 가질 수 있습니다.

- 채권(Fixed Income)
  - 정부나 기업이 자금을 빌릴 때 발행.
  - 채권을 매수하는 사람은 채무자에게 돈을 빌려주며, 정해진 이자와 함께 돌려받음

- 통화(Currencies, FX)
  - 여러 통화를 사고파는 외환 시장. 환율은 경제 상황, 금리, 정치 안정성 등의 요인에 영향을 받음.

- 원자재(Commodities)
  - 금, 석유, 농산물 등 물리적인 자산. 인플레이션에 대한 헷지 수단이나 세계 경제 흐름을 반영한 수익 창출 수단으로 활용

- 대체 투자(Alternative Investments)
  - 부동산, 사모펀드, 헤지펀드 등. 전통적 자산보다 높은 수익률과 분산 투자 효과를 줄 수 있지만, 유동성이 낮고 평가가 어렵다는 단점이 있음

- 시장 미시구조(Market Microstructure)
  - 시장 미시구조는 금융 시장에서의 **증권 거래 방식을 연구하는 분야**
    - 주문 흐름(order flow)
    - 유동성(liquidity) : 자산을 가격 변동 없이 얼마나 쉽게 사고팔 수 있는지를 나타내는 지표
    - 시장 효율성(market effectiveness)
    - 가격 발견(price discovery) : 시장 내에서 자산의 가격이 어떻게 결정되는지를 설명하는 과정
  
> **시장 미시구조 개념**은 모델링이나 알고리즘 트레이딩, 리스크 관리, 거래 전략 수립 등에 있어 핵심적인 통찰을 제공

### Applications of Data Science in Finance
- 데이터 과학을 통해 금융을 개선하려는 시도가 급증
  - 시장 방향 예측
  - 금융 사기 탐지
    - 짧은 시간 안에 반복적으로 소액 결제가 발생하는 경우
    - 특정 상점에서 반복적으로 고액 결제가 발생하는 경우
  - 리스크 관리
    - 포트폴리오에 영향을 줄 수 있는 잠재적 위험 요소를 식별
  - 신용 점수 평가
    - 개인 또는 기업의 재무 데이터와 신용 기록을 분석하여, 신용도를 예측하고 대출 여부를 결정
  - 자연어 처리 (NLP)
    - 뉴스, 리포트, 소셜 미디어 등 비정형 금융 데이터를 분석하여 의사결정에 도움이 되는 통찰을 추출

### Summary
- 데이터 과학은 매일 발전하고 있으며, 데이터 해석 능력을 향상시키기 위한 새로운 기법과 모델들이 지속적으로 등장