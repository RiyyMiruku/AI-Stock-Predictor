import yfinance as yf
import datetime

def is_today_market_closed(symbol: str) -> bool:
  
    today = datetime.datetime.now().date()
    data = yf.download(symbol, start=today, end=today + datetime.timedelta(days=1), progress=False)

    if data.empty:
        return False
    else:
        return True
    

if __name__ == '__main__':  
    symbol = "^TWII"
    # closed = is_today_market_closed(symbol)
    # print(f"Is the market closed for {symbol} today? {closed}")

    today = datetime.datetime.now().date()
    df = yf.download(symbol, start=today+datetime.timedelta(days=-1), end=today , progress=False)

    print(df.index[0].date() != today+datetime.timedelta(days=-2))
    print(today+datetime.timedelta(days=-2))
    print(df.index[0].date())
