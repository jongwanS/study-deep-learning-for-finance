# deep-learning-for-finance
This is the official repository for the book "Deep Learning for Finance"


# 파이썬 가상환경 설정
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 사용후 가상환경 빠져나오기
deactivate

# pip lib 설치
pip install pandas_datareader pandas

# requiement 생성
pip freeze > requirements.txt

# 실행
````python
python3 ./Chapter\ 1/VIX_Analysis.py
````