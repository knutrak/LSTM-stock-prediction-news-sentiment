import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
from transformers import pipeline


classifier = pipeline('text-classification', model='ProsusAI/finbert')

def get_sentiment(text):
    result = classifier(text)
    return result[0]['label'], result[0]['score']


STOCK = "Apple"
ticker = "aapl"
columns = ['datetime', 'title', 'source', 'link', 'top sentiment', 'sentiment score']
CSV_PATH = fr'C:\RoadToMlGod\stock_predictor\data\articles_{STOCK}.csv'


if os.path.exists(CSV_PATH):
    df_existing = pd.read_csv(CSV_PATH)
    existing_pairs = set(zip(df_existing['title'], df_existing['datetime']))
    df = df_existing.copy()
else:
    df = pd.DataFrame(columns=columns)
    existing_pairs = set()


counter = 0
page = 1
# max_pages = 3  # for testing; remove or increase later

while True:
    url = f'https://markets.businessinsider.com/news/{ticker}-stock?p={page}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    articles = soup.find_all('div', class_='latest-news__story')

    if not articles:
        break

    new_rows = []
    for article in articles:
        try:
            datetime = article.find('time', class_='latest-news__date').get('datetime')
            title = article.find('a', class_='news-link').text.strip()
            source = article.find('span', class_='latest-news__source').text.strip()
            link = article.find('a', class_='news-link').get('href').strip()
        except AttributeError:
            continue

        pair = (title, datetime)
        if pair in existing_pairs:
            continue  

        # Sentiment analysis
        top_sentiment, sentiment_score = get_sentiment(title)

        new_rows.append([datetime, title, source, link, top_sentiment, sentiment_score])
        existing_pairs.add(pair)
        counter += 1

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=columns)
        df = pd.concat([df, new_df], ignore_index=True)
        print(f'Page {page} scraped | New articles this page: {len(new_rows)} | Total new so far: {counter}')

        page += 1
    else:
        break
    
    # if page > max_pages:
    #     break


df.to_csv(CSV_PATH, index=False)
print(f"Scraping complete. Added {counter} new unique articles. Total in file: {len(df)}")
