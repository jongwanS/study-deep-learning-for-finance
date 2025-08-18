import requests
import pandas as pd
import time
import json

# 1. Binance API 데이터 가져오기
def get_binance_data(symbol="BTCUSDT", interval="1d", limit=500):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df[["timestamp","close","volume"]]

# 2. 이동평균선 계산
def add_moving_averages(df):
    df["MA50"] = df["close"].rolling(window=50).mean()
    df["MA200"] = df["close"].rolling(window=200).mean()
    df["MA400"] = df["close"].rolling(window=400).mean()
    return df

# 3. Slack 메시지 전송
def send_slack_message(webhook_url, text):
    payload = {"text": text}
    requests.post(webhook_url, data=json.dumps(payload))
# 4. 실행 루프 (여러 심볼 + interval 지정 가능)
def run_bot(symbols, interval="1d", sleep_seconds=60*60*24):
    SLACK_WEBHOOK = ""  # 👉 본인 Webhook URL 넣기

    while True:
        all_msgs = []  # 심볼별 메시지를 모아놓을 리스트

        for symbol in symbols:
            try:
                df = get_binance_data(symbol=symbol, interval=interval)
                df = add_moving_averages(df)
                latest = df.iloc[-1]

                # 이동평균 차이 계산
                diff_50_200 = latest["MA50"] - latest["MA200"]
                diff_50_400 = latest["MA50"] - latest["MA400"]
                diff_200_400 = latest["MA200"] - latest["MA400"]

                msg = (
                    f"📊 {symbol} ({interval})\n"
                    f"현재가: {latest['close']}\n"
                    f"50일-200일: {diff_50_200:.2f}\n"
                    f"50일-400일: {diff_50_400:.2f}\n"
                    f"200일-400일: {diff_200_400:.2f}\n"
                )
                all_msgs.append(msg)

            except Exception as e:
                all_msgs.append(f"⚠️ {symbol} 데이터 처리 오류: {e}")

        # 모든 심볼 결과를 하나의 메시지로 합쳐서 전송
        final_message = "\n----------------------\n".join(all_msgs)
        send_slack_message(SLACK_WEBHOOK, final_message)

        # 지정한 interval만큼 대기
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    # 👉 원하는 심볼과 주기를 여기서 지정
    symbols = ["BTCUSDT", "ETHUSDT"]  # 여러 개 가능
    run_bot(symbols, interval="4h", sleep_seconds=60*60*24)  # interval: Binance 기준, sleep_seconds: 체크 주기
