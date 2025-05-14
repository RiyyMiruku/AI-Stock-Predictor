import requests
from bs4 import BeautifulSoup

def fetch_news():
    url = "https://tw.stock.yahoo.com/news"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    articles = soup.select('li.js-stream-content')[:5]
    news = []
    for art in articles:
        title = art.text.strip()
        link = art.a['href'] if art.a else None
        news.append({"title": title, "link": link})
    return news
