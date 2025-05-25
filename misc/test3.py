import yfinance as yf
from datetime import datetime, timedelta

def is_today_market_closed(symbol: str) -> bool:
  
    today = datetime.now().date()
    data = yf.download(symbol, start=today, end=today + timedelta(days=1), progress=False)

    #因為yfinance如果在非交易日下載會回傳前一天的資料，所以需要檢查第一筆資料的日期是否為今天

    #若下載的筆數非空，判斷下載的資料的日期是否為今天
    if not data.index.empty:
        # 若第一筆資料的日期為今天，表示今天是交易日
        if data.index[0].date() == today:
            return True
    return False

if __name__ == '__main__':  
    symbol = "^TWII"
    # closed = is_today_market_closed(symbol)
    # print(f"Is the market closed for {symbol} today? {closed}")

    print(is_today_market_closed(symbol))
