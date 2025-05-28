import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TICKER = "^TWII"
NEWS_COUNT = 10
MODEL_PATH = "model/simple_factor_model.joblib"
PENDING_NEWS_FILE = "data/pending_news.json"
TRAINED_UNTIL_PATH = "model/trained_until.txt"
SCALER_PATH = "model/scaler.save"
FIRST_TRAIN_FLAG = "model/first_train_done.flag"