from datetime import datetime, timedelta
import yfinance as yf


def is_today_market_open(symbol: str) -> bool:
    """以 yfinance 判斷今天是否為該標的的交易日。

    yfinance 在非交易日會回傳前一天的資料，因此須比對日期是否為今天。
    """
    today = datetime.now().date()
    data = yf.download(
        symbol, start=today, end=today + timedelta(days=1), progress=False
    )

    if not data.index.empty and data.index[0].date() == today:
        return True
    return False
