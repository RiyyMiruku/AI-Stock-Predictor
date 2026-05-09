import yfinance as yf
import pandas as pd


def calc_rsi(series, period: int = 14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_technical_factors(ticker: str) -> pd.DataFrame:
    """計算最近一日的技術指標（動能、波動率、均線、乖離率、RSI）。"""
    df = yf.download(ticker, period="14d", interval="1d")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df["momentum_1d"] = df["Close"].pct_change()
    df["volatility_5d"] = df["Close"].rolling(5).std()
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["bias_5"] = (df["Close"] - df["ma_5"]) / df["ma_5"]
    df["rsi_5"] = calc_rsi(df["Close"], 5)

    print("今日日期：", df.index[-1], "收盤價：", df["Close"].iloc[-1])

    return df[["momentum_1d", "volatility_5d", "ma_5", "bias_5", "rsi_5"]].dropna().tail(1)
