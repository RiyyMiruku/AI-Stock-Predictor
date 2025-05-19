from dotenv import dotenv_values
from groq import Groq


GROQ_API_KEY = dotenv_values().get("GROQ_API_KEY")
# print(GROQ_API_KEY)
client = Groq(api_key=GROQ_API_KEY)

def analyze_with_groq( new : dict[str, str] ,ticker)-> str:
    prompt = f"""
你是一位專業的台股投資分析師，請根據以下新聞內容，完全按照下列格式回傳對台股「{ticker}」的影響指標，不要其他描述。
{{
  "sentiment_score": 0~100,
  "volatility_hint": 0~100,
  "confidence_level": 0~100,
  "aggregated_signal_score": 0~100,
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
    return chat_completion.choices[0].message.content

# print(analyze_with_groq({"title":"test","content":"test"},"^TWII"))
