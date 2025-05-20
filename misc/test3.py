import yfinance as yf
import datetime

def get_today_close_price(symbol: str) -> bool:
  
    today = datetime.datetime.now().date()
    data = yf.download(symbol, start=today, end=today + datetime.timedelta(days=1), progress=False)

    if data.empty:
        return False
    else:
        return True



