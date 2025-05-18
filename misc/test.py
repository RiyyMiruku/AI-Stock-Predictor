# import os
# import requests
from dotenv import dotenv_values
from groq import Groq


GROQ_API_KEY = dotenv_values().get("GROQ_API_KEY")
print(GROQ_API_KEY)
client = Groq(api_key=GROQ_API_KEY)

def analyze_with_groq( new : dict[str, str] )-> str:
    prompt = """
你是專業台股投資分析師，將指標以以下格式回評估新聞影響，僅回答數字，不須文字分析：
{sentiment_score: 0~1, 
volatility_hint: 0~1, 
confidence_level: 0~1, 
aggregated_signal_score: 0~1, 
topic_type: earnings / rumor / macro}
"""+f'{new['content']}'
    
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

print(analyze_with_groq({'content':'時報-台北電】美國總統川普16日在阿布達比舉行的商業圓桌會議上表示，將 在「未來兩到三周」，單方面對各國訂出新的關稅稅率，他說，美國政府沒有足夠時間與所有貿易夥伴達成協議。這番言論，暗示他對貿易談判進展太慢感到挫敗。\n川普說，儘管有「150個國家」希望達成協議，但「我們不可能滿足所有希 望見 到我們的國家」。川普政府先前曾把在4月2日「解放日」對多數國家宣布的對等關稅削減至10％、為期90天。\n他 補充說，接下來的兩到三周內，財長貝森特與商務部長霍華德盧特尼克將會「發出信函，基本上告知人們，在美國做生意需要支付多少錢」。\n迄今為止，川普政府已成功宣布兩項新的貿易談判框架，分別是與英國及中國達成的協議。\n川普談判團隊指出，他們目前正與十幾個國家積極協商。美國政府先前曾透露，即將與印度及日本達成協議框架，至於韓國因為六月將舉行總統大選，談判因而遭到延後。\n雖然盧特尼克與部分政府官員稍早曾把10％稅率描述為「基準」關稅，但遭川普否認，認為稅率還要更高。惠譽信評機構預期，在90天對等關稅暫停期於7月8日到期後，美國仍可能對進口商品維持13％的平均關稅稅率。(新聞來源 : 工商時報一蕭麗君／綜合外電報導'}))