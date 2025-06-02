from dotenv import dotenv_values
from groq import Groq
import json

GROQ_API_KEY = dotenv_values().get("GROQ_API_KEY")
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
你是一位專業的台股投資分析師，請根據以下新聞內容，以json格式回傳該新聞對台股「{ticker}」的影響指標，不要其他描述。
{{
  "sentiment_score": 0~100(極端負面~極端正面),
  "volatility_hint": 0~100(暗示市場平穩~暗示市場可能有極大波動),
  "confidence_level": 0~100(你對於這篇新聞分析的把握程度；若此新聞與股市不相關，請將此項訂為0),
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
        response_format={"type": "json_object"},
        temperature=0
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

# print(analyze_with_groq({"title":"test","content":'''【時報編譯柳繼剛綜合外電報導】美國總統川普在自己的社交平台Truth Social上發文說，要把進口到美國的鋼鐵以及鋁製品關稅，提高一倍到50%，且將於六月4日生效，包括韓國與越南等主要輸美的亞洲煉鋼廠，股價在二日交易時，大部分都大幅走低。

# 收盤時，韓國最大的鋼鐵業者浦項鋼鐵（POSCO）大跌2.4%，現代汽車集團旗下的現代鋼鐵（Hyundai Steel）也大跌2.7%。同時，SeAH鋼鐵一度更是狂跌18%，跌幅最後收斂在8%。

# 在港交所掛牌的中國大陸國有煉鋼廠中，重慶鋼鐵大跌2.3%，馬鞍山鋼鐵收跌1.1%，鞍鋼也跌1.3%，恆生指數二日跌0.7%左右。同時，中國鋁業跌0.2%，中國宏橋也收跌0.57%。

# 以韓國來說，鋼鐵業者表示，50%的關稅將為韓國鋼鐵出口業者帶來更多挑戰。美國鋼鐵價格已上漲，但為了避免受到美國政府的審查，其實韓國出口商一直沒有大幅增加輸美的產品。

# 業者坦言，如果美國鋼鐵價格不再繼續上漲，出口商要承擔的成本會很高。韓國政府二日表示，已跟浦項鋼鐵以及現代鋼鐵在內，韓國幾個大型煉鋼廠的高層開過緊急會議。

# 不過，BlueScope在澳洲證交所（ASX）的股價，早盤一度飆升近10%，最後收盤也大漲4.4%，因為這家澳洲的鋼鐵巨擘在北美有好幾個鋼鐵事業，包括位於俄亥俄州的北極星（North Star）煉鋼廠。'''},"^TWII"))
