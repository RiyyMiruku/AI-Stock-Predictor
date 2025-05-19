import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TICKER = "^TWII"
NEWS_COUNT = 10
MODEL_PATH = "data/simple_factor_model.joblib"
PENDING_NEWS_FILE = "data/pending_news.json"