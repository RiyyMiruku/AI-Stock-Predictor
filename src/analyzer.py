import json
from groq import Groq
from src.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)


def sentiment_to_num(x):
    if x == "positive":
        return 1
    elif x == "neutral":
        return 0
    elif x == "negative":
        return -1
    return None


def analyze_with_groq(news: dict[str, str], ticker: str) -> dict | None:
    """用 LLM 將單則新聞轉成數值化的投資因子。"""
    prompt = f"""
你是一位專業的台股投資分析師，請根據以下新聞內容，以json格式回傳該新聞對台股「{ticker}」的影響指標，不要其他描述。
{{
  "sentiment_score": 0~100(對市場的影響力小~對市場的影響力大),
  "volatility_hint": 0~100(暗示市場平穩~暗示市場可能有極大波動),
  "confidence_level": 0~100(你對於這篇新聞分析的把握程度；若此新聞與股市不相關，請將此項訂為0),
  "positive_neutral_negative": "positive" 或 "neutral" 或 "negative"
}}
新聞:{news['content']}
"""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        response_format={"type": "json_object"},
        temperature=0,
    )
    answer = chat_completion.choices[0].message.content

    try:
        new_score = json.loads(answer)
        if "positive_neutral_negative" in new_score:
            new_score["positive_neutral_negative"] = sentiment_to_num(
                new_score["positive_neutral_negative"]
            )
    except json.JSONDecodeError:
        print("⚠️ 無法解析 JSON 格式")
        new_score = None
    return new_score
