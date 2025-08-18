import os
import time
import json
import math
import requests
import pandas as pd
import numpy as np

# ===============================
# Binance 데이터
# ===============================
def get_binance_data(symbol="BTCUSDT", interval="1d", limit=600):
    """
    Binance Klines 가져오기 (종가/거래량/타임스탬프)
    limit=600: 지표 계산(최대 MA400) 여유분 확보
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df[["timestamp","close","volume"]]

# ===============================
# 보조지표 계산
# ===============================
def add_ma(df):
    df["MA50"]  = df["close"].rolling(50).mean()
    df["MA200"] = df["close"].rolling(200).mean()
    df["MA400"] = df["close"].rolling(400).mean()
    return df

def add_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    df["MACD"] = macd_line
    df["MACD_SIGNAL"] = signal_line
    df["MACD_HIST"] = hist
    return df

def add_bollinger(df, window=20, num_std=2):
    ma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    df["BB_MID"] = ma
    df["BB_UP"] = ma + num_std*std
    df["BB_LOW"] = ma - num_std*std
    return df

def add_volume_stats(df, window=20):
    df["VOL_MA"] = df["volume"].rolling(window).mean()
    return df

def enrich(df):
    df = add_ma(df)
    df = add_rsi(df, 14)
    df = add_macd(df, 12, 26, 9)
    df = add_bollinger(df, 20, 2)
    df = add_volume_stats(df, 20)
    return df

# ===============================
# 전략 시그널
# ===============================
def strategy_signals(df):
    """
    3가지 전략의 Buy/Sell 시그널과 근거 텍스트 리턴
    - 전략1: RSI + MA200 + 거래량
    - 전략2: MACD 크로스
    - 전략3: 볼린저 밴드 돌파 + 거래량
    """
    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    price = latest["close"]
    rsi = latest["RSI"]
    ma200 = latest["MA200"]
    vol = latest["volume"]
    vol_ma = latest["VOL_MA"]

    macd = latest["MACD"]; macd_sig = latest["MACD_SIGNAL"]
    macd_prev = prev["MACD"]; macd_sig_prev = prev["MACD_SIGNAL"]

    bb_up = latest["BB_UP"]; bb_low = latest["BB_LOW"]

    buy_signals = []
    sell_signals = []
    notes = []

    # ===== 전략 1: RSI + MA200 + 거래량
    # Buy: RSI<30 AND (가격<=MA200*1.01) AND 거래량>=VOL_MA
    if pd.notna(rsi) and pd.notna(ma200) and pd.notna(vol_ma):
        if (rsi < 30) and (price <= ma200 * 1.01) and (vol >= vol_ma):
            buy_signals.append("전략1(BUY): 과매도+MA200 부근 지지+거래량 확증")
        # Sell: RSI>70 (과열). 보수적으로 MA200 멀리 위(>1.05)면 가중치↑
        if rsi > 70:
            cond = "전략1(SELL): 과매수"
            if ma200 and price >= ma200 * 1.05:
                cond += " (+MA200 대비 과열)"
            sell_signals.append(cond)

    # ===== 전략 2: MACD 크로스
    # Buy: MACD 크로스 업
    # MACD는 일반적으로 **단기 지수이동평균(EMA)**에서 **장기 지수이동평균(EMA)**을 뺀 값
    if pd.notna(macd) and pd.notna(macd_sig) and pd.notna(macd_prev) and pd.notna(macd_sig_prev):
        if (macd_prev < macd_sig_prev) and (macd > macd_sig):
            buy_signals.append("전략2(BUY): MACD 골든크로스")
        if (macd_prev > macd_sig_prev) and (macd < macd_sig):
            sell_signals.append("전략2(SELL): MACD 데드크로스")

    # ===== 전략 3: 볼린저 돌파 + 거래량
    # Buy: 종가가 상단 돌파 AND 거래량 스파이크(>=1.2 * VOL_MA)
    # Sell: 종가가 하단 이탈 AND 거래량 스파이크
    if pd.notna(bb_up) and pd.notna(bb_low) and pd.notna(vol_ma):
        if (price > bb_up) and (vol >= 1.2 * vol_ma):
            buy_signals.append("전략3(BUY): BB 상단 돌파 + 거래량")
        if (price < bb_low) and (vol >= 1.2 * vol_ma):
            sell_signals.append("전략3(SELL): BB 하단 이탈 + 거래량")


    # ===== 보조: 이동평균 간 차이 (계속 보고용)
    ma50_prev = prev["MA50"];
    ma200_prev = prev["MA200"];
    ma400_prev = prev["MA400"];
    ma50 = latest["MA50"];
    ma200 = latest["MA200"];
    ma400 = latest["MA400"];
    cross_50_200 = check_cross(ma50_prev, ma200_prev, ma50, ma200)
    cross_50_400 = check_cross(ma50_prev, ma400_prev, ma50, ma400)
    cross_200_400 = check_cross(ma200_prev, ma400_prev, ma200, ma400)

    diffs_text = ""
    if pd.notna(ma50) and pd.notna(ma200):
        diffs_text += f"50-200: {ma50 - ma200:.2f} \n"
    if pd.notna(ma50) and pd.notna(ma400):
        diffs_text += f"50-400: {ma50 - ma400:.2f} \n"
    if pd.notna(ma200) and pd.notna(ma400):
        diffs_text += f"200-400: {ma200 - ma400:.2f}"
    if cross_50_200: notes.append(f"50 vs 200: {cross_50_200}")
    if cross_50_400: notes.append(f"50 vs 400: {cross_50_400}")
    if cross_200_400: notes.append(f"200 vs 400: {cross_200_400}")
    # ===== 최종 의사결정(합의)
    buy_count = len(buy_signals)
    sell_count = len(sell_signals)
    if buy_count > sell_count:
        decision = "BUY"
    elif sell_count > buy_count:
        decision = "SELL"
    else:
        decision = "NEUTRAL"

    # 메모(맥락)
    notes.append(f"RSI: {rsi:.2f}" if pd.notna(rsi) else "RSI: n/a")
    if pd.notna(ma200):
        notes.append(f"MA200乂乂거리: {(price/ma200-1)*100:.2f}%")
    if pd.notna(vol_ma):
        notes.append(f"Vol/VolMA: {vol/vol_ma:.2f}x")

    return {
        "price": price,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "decision": decision,
        "diffs_text": diffs_text.strip(),
        "notes": " \n".join(notes)
    }

# ===============================
# Slack
# ===============================
def send_slack_message(text, webhook_url=None):
    url = webhook_url or os.getenv("SLACK_WEBHOOK")
    if not url:
        raise RuntimeError("SLACK_WEBHOOK 환경변수를 설정하세요.")
    payload = {"text": text, "mrkdwn": True}
    response = requests.post(url, data=json.dumps(payload), timeout=10)
    print(response.status_code, response.text)

# ===============================
# 실행 루프: 여러 심볼 + 인터벌 지정 + 한 번에 하나의 메시지
# ===============================
def run_bot(symbols, interval="1d", sleep_seconds=60*60*24, webhook_url=None):
    while True:
        all_msgs = []
        for sym in symbols:
            try:
                df = get_binance_data(symbol=sym, interval=interval, limit=600)
                df = enrich(df).dropna().reset_index(drop=True)
                if len(df) < 5:
                    all_msgs.append(f"⚠️ {sym}: 데이터 부족")
                    continue

                sig = strategy_signals(df)

                # 심볼별 블록 메시지
                block = [
                    f"📊 *{sym}* ({interval})",
                    f"*가격*: {sig['price']:.4f}",
                    f"*메모*: \n"
                    f"{sig['notes']}",
                    f"*MA차이*: \n"
                    f"{sig['diffs_text']}" if sig['diffs_text'] else "",
                    "",
                ]
                if sig["buy_signals"]:
                    block.append("✅ BUY 신호:")
                    block += [f"• {s}" for s in sig["buy_signals"]]
                if sig["sell_signals"]:
                    block.append("❌ SELL 신호:")
                    block += [f"• {s}" for s in sig["sell_signals"]]
                block.append(f"➡️ 최종 판단: *{sig['decision']}*")

                all_msgs.append("\n".join([b for b in block if b != ""]))

            except Exception as e:
                all_msgs.append(f"⚠️ {sym} 처리 오류: {e}")

        final_message = "\n" + ("\n" + ("-"*28) + "\n").join(all_msgs)
        try:
            send_slack_message(final_message, webhook_url=webhook_url)
        except Exception as e:
            print("Slack 전송 실패:", e)

        time.sleep(sleep_seconds)

# MA 교차 판별
def check_cross(prev_short, prev_long, curr_short, curr_long):
    if pd.notna(prev_short) and pd.notna(prev_long) and pd.notna(curr_short) and pd.notna(curr_long):
        if prev_short < prev_long and curr_short > curr_long:
            return "💹 골든크로스"
        elif prev_short > prev_long and curr_short < curr_long:
            return "🔻 데드크로스"
    return None

if __name__ == "__main__":
    # ▶ 원하는 심볼/인터벌/주기 설정
    symbols = ["BTCUSDT"]
    #symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
    # Binance interval 예: 1m, 5m, 15m, 1h, 4h, 1d, 1w
    run_bot(symbols, interval="4h", sleep_seconds=60*60*24)