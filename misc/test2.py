import yfinance as yf
from datetime import datetime

ticker = "^TWII"
today = datetime.today().strftime("%Y-%m-%d")

df = yf.download(ticker, period="2d", interval="1d")
print(df)

if today in df.index.strftime("%Y-%m-%d"):
    close = df.loc[df.index.strftime("%Y-%m-%d") == today, "Close"].values[0]
    print(f"{today} 台股收盤價：{close}")
else:
    print(f"{today} 台股收盤價尚未公布")