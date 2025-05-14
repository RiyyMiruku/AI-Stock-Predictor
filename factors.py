import yfinance as yf
import pandas as pd

def compute_technical_factors(ticker):
    df = yf.download(ticker, period="7d", interval="1d")
    df["momentum_1d"] = df["Close"].pct_change()
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(5).mean()
    df["volatility_5d"] = df["Close"].rolling(5).std()
    return df[["momentum_1d", "volume_ratio", "volatility_5d"]].dropna()
