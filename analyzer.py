from dotenv import dotenv_values
from groq import Groq
import json

GROQ_API_KEY = dotenv_values().get("GROQ_API_KEY")
# print(GROQ_API_KEY)
client = Groq(api_key=GROQ_API_KEY)

def sentiment_to_num(x):
    if x == "positive":
        return 1
    elif x == "neutral":
        return 0
    elif x == "negative":
        return -1
    else:
        return None
    
def analyze_with_groq( new : dict[str, str] ,ticker)-> str:
    prompt = f"""
你是一位專業的台股投資分析師，請根據以下新聞內容，完全按照下列格式回傳對台股「{ticker}」的影響指標，不要其他描述。
{{
  "sentiment_score": 0~100,
  "volatility_hint": 0~100,
  "confidence_level": 0~100,
  "positive_neutral_negative": "positive" 或 "neutral" 或 "negative"
}}
新聞:{new['content']}
"""
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.1-8b-instant",
    )

    answer = chat_completion.choices[0].message.content

    # 將 positive/neutral/negative 轉換為 1/0/-1
    try:
        new_score = json.loads(answer)
        if "positive_neutral_negative" in new_score:
            new_score["positive_neutral_negative"] = sentiment_to_num(new_score["positive_neutral_negative"])
    except json.JSONDecodeError:
        print("⚠️ 無法解析 JSON 格式")
        new_score = None
    return new_score

# print(analyze_with_groq({"title":"test","content":"test"},"^TWII"))
