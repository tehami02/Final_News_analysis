from flask import Flask, render_template, request, redirect, url_for, session
import requests
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from textblob import TextBlob

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# CSV URL
csv_url = 'https://raw.githubusercontent.com/tehami02/News_Analysis_NEW/main/Datanews4.csv'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    # No need to get the source now; just redirect to display news
    return redirect(url_for('display_news'))

@app.route('/display_news')
def display_news():
    # Read the CSV file
    try:
        df = pd.read_csv(csv_url)
        articles = df.head(8).to_dict(orient='records')  # Get top 8 articles
    except Exception as e:
        return render_template('error.html', message="Error loading CSV data: " + str(e))

    # Pass articles to the template
    return render_template('display_news.html', articles=articles)

@app.route('/analyze')
def analyze():
    heading = request.args.get('heading')  # Get the heading from the query parameters
    print(heading)
    try:
        df = pd.read_csv(csv_url)  # Read the CSV file
        article = df[df['heading'] == heading].iloc[0]  # Get the corresponding row based on the heading
        
        article_content = article['data']  # Get the data for that heading
    except Exception as e:
        return render_template('error.html', message=str(e))
    
    if not article_content:
        return render_template('error.html', message="No content found to analyze.")

    # Continue with your analysis logic...
    try:
        vectorizer = CountVectorizer(stop_words='english')
        bow_matrix = vectorizer.fit_transform([article_content])
        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(bow_matrix)

        bow_data = dict(zip(vectorizer.get_feature_names_out(), bow_matrix.toarray()[0]))
        tfidf_data = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))

        top_bow_words = sorted(bow_data.items(), key=lambda item: item[1], reverse=True)[:5]
        top_tfidf_words = sorted(tfidf_data.items(), key=lambda item: item[1], reverse=True)[:5]

        # Perform sentiment analysis
        blob = TextBlob(article_content)
        sentiment_score = blob.sentiment.polarity
        sentiment_label = "Positive" if sentiment_score > 0.1 else "Negative" if sentiment_score < 0 else "Neutral"

        return render_template('analysis_news.html', article_content=article_content, bow_data=bow_data, tfidf_data=tfidf_data, top_bow_words=top_bow_words, top_tfidf_words=top_tfidf_words, sentiment_score=sentiment_score, sentiment_label=sentiment_label)
    except Exception as e:
        return render_template('error.html', message=str(e))


if __name__ == "__main__":
    app.run(debug=True)
