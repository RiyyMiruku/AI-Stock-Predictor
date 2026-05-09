import requests
from bs4 import BeautifulSoup
import feedparser


def fetch_yahoo_finance_news(news_count: int) -> list[dict[str, str]]:
    """抓取 Yahoo 財經 RSS 新聞的標題、連結與內文。"""
    rss_url = "https://tw.stock.yahoo.com/rss"
    feed = feedparser.parse(rss_url)
    news = []
    for entry in feed.entries[:news_count]:
        title = entry.title
        link = entry.link
        try:
            resp = requests.get(link, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            article = soup.find("article")
            if article:
                paragraphs = [p.get_text(strip=True) for p in article.find_all("p")]
                content = "\n".join(paragraphs)
            else:
                content = ""
        except Exception:
            content = ""
        news.append({"title": title, "link": link, "content": content})
    return news
