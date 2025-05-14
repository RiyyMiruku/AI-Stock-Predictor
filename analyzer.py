import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_article(title):
    prompt = f"""
    以下為一則股市新聞標題，請依照以下格式回傳分析：
    - sentiment_score: 介於 -1 到 1
    - volatility_hint: 0 或 1
    - topic_type: earnings / rumor / macro
    
    標題：{title}
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
