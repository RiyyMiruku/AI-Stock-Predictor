import os
from dotenv import load_dotenv

load_dotenv()

# 預測標的（Yahoo Finance ticker）
TICKER = "^TWII"

# 每日抓取的新聞數量
NEWS_COUNT = 10

# 路徑設定
DATA_DIR = "data"
MODEL_DIR = "model"

FEATURES_FILE = os.path.join(DATA_DIR, f"{TICKER}_daily_features.csv")
PENDING_NEWS_FILE = os.path.join(DATA_DIR, "pending_news.json")

MODEL_PATH = os.path.join(MODEL_DIR, "simple_factor_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.save")
TRAINED_UNTIL_PATH = os.path.join(MODEL_DIR, "trained_until.txt")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
