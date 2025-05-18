import requests
from bs4 import BeautifulSoup
import feedparser

def fetch_yahoo_finance_news(NEWS_COUNT : int)-> list[dict[str, str]]:
    """
    抓取 Yahoo 財經新聞標題、連結與內文
    """
    rss_url = "https://tw.stock.yahoo.com/rss"
    feed = feedparser.parse(rss_url)
    news = []
    for entry in feed.entries[:NEWS_COUNT]:
        title = entry.title
        link = entry.link
        # 取得新聞內文
        try:
            resp = requests.get(link, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            # Yahoo 財經新聞內文通常在 <article> 標籤
            article = soup.find("article")
            if article:
                paragraphs = [p.get_text(strip=True) for p in article.find_all("p")]
                content = "\n".join(paragraphs)
            else:
                content = ""
        except Exception as e:
            content = ""
        news.append({"title": title, "link": link, "content": content})
    return news
# print(fetch_yahoo_finance_news(3))