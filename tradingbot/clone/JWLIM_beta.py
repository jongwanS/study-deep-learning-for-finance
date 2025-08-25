import os
import requests
import json
from dotenv import load_dotenv

#.env load
load_dotenv()
slack_url = os.getenv("SLACK_WEBHOOK_URL")

# def map_kline_data(kline_data_list):
#     """
#     K-line 데이터 리스트를 받아 딕셔너리로 매핑하여 반환합니다.
#     """
#     # 데이터 항목의 이름을 키로 정의합니다.
#     data_keys = [
#         "open_time",
#         "open_price",
#         "high_price",
#         "low_price",
#         "close_price",
#         "volume",
#         "close_time",
#         "quote_asset_volume",
#         "number_of_trades",
#         "taker_buy_base_volume",
#         "taker_buy_quote_volume",
#         "unused"
#     ]
#
#     # zip 함수를 이용해 키와 값을 묶어 딕셔너리로 만듭니다.
#     # 불필요한 "unused" 필드는 제거합니다.
#     kline_dict = {key: value for key, value, in zip(data_keys, kline_data_list) if key != "unused"}
#
#     # 일부 값은 숫자로 변환합니다.
#     kline_dict['open_price'] = float(kline_dict['open_price'])
#     kline_dict['high_price'] = float(kline_dict['high_price'])
#     kline_dict['low_price'] = float(kline_dict['low_price'])
#     kline_dict['close_price'] = float(kline_dict['close_price'])
#     kline_dict['volume'] = float(kline_dict['volume'])
#     kline_dict['quote_asset_volume'] = float(kline_dict['quote_asset_volume'])
#     kline_dict['taker_buy_base_volume'] = float(kline_dict['taker_buy_base_volume'])
#     kline_dict['taker_buy_quote_volume'] = float(kline_dict['taker_buy_quote_volume'])
#
#     return kline_dict

# 1. binance data 가져오기
def get_binance_data(symbol="BTCUSDT", interval="1d", limit=500):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": 1}
    response = requests.get(url, params=params)
    data = response.json()
    return data;

def run_bot(symbols):
    for sym in symbols:
        kline_data_list = get_binance_data(sym)
        # map_kline_data(kline_data_list)
    pass


if __name__ == "__main__":
    # ▶ 원하는 심볼/인터벌/주기 설정
    #symbols = ["BTCUSDT"]
    #symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
    symbols = ["BTCUSDT", "ETHUSDT",]
    # Binance interval 예: 1m, 5m, 15m, 1h, 4h, 1d, 1w
    run_bot(symbols)

