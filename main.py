from __future__ import annotations
import os
import re
import asyncio
import csv
import pickle
import numpy as np
import nltk
import torch
import mysql.connector
import spacy
import smtplib
import pandas as pd
import pymysql
import emoji
import logging
import time
from time import sleep
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from mysql.connector import Error
from typing import NoReturn
from twikit import Client
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy import text


# Import configurations
from config import (BERT_SCALER_PATH, SCALER_PATH, XGB_MODEL_PATH)

# Initialize NLTK resources
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('vader_lexicon')
#nltk.download('punkt_tab')

# Load pre-trained tools
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
sia = SentimentIntensityAnalyzer()

# Global Objects
tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Load Scalers and Models
try:
    with open(BERT_SCALER_PATH, 'rb') as f:
        bert_scaler = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(XGB_MODEL_PATH, 'rb') as f:
        loaded_rf_model = pickle.load(f)
    print("Models and scalers loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    raise

# Twitter Client Initialization
client = Client('en-US')

import mysql.connector
from mysql.connector import Error

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',  # Replace with your MySQL host
            user='',  # Replace with your MySQL username
            password='',  # Replace with your MySQL password
            database=''  # Replace with your MySQL database name
        )
        if connection.is_connected():
            print("Connected to MySQL database")
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        raise

def is_tweet_in_database(connection, tweet_id: int) -> bool:
    query = "SELECT EXISTS(SELECT 1 FROM raw_tweets WHERE tweet_id = %s)"
    cursor = connection.cursor()
    cursor.execute(query, (tweet_id,))
    exists = cursor.fetchone()[0]
    cursor.close()
    return bool(exists)


# Text Preprocessing
def clean_text(text: str) -> str:
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s#]', '', text)  # Remove non-alphanumeric characters except hashtags
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.lower().strip()
    return text

def preprocess_text(text: str) -> str:
    text = clean_text(text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in STOPWORDS and not word.startswith('#')]
    return ' '.join(tokens)

# Extract Hashtags
def extract_hashtags(text: str) -> str:
    hashtags = re.findall(r'#\w+', text)
    return ", ".join(hashtags)

# Extract Keywords
def extract_keywords(text: str) -> str:
    clean = clean_text(text)
    tokens = tweet_tokenizer.tokenize(clean)
    keywords = [word for word in tokens if word not in STOPWORDS and not word.startswith('#')]
    return ", ".join(keywords)

def extract_brands(text):
    """
    Extract brand names (ORG entities) from text using spaCy NER.

    Parameters:
    - text (str): Input text containing brand names.

    Returns:
    - List of extracted brand names
    """
    doc = nlp(text)
    brands = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return ", ".join(brands)
    

# Extract BERT Features
def extract_bert_features_batch(texts, batch_size=32) -> np.ndarray:
    all_features = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_features = outputs.last_hidden_state[:, 0, :].detach().numpy()
        all_features.extend(batch_features)
    return np.array(all_features)

# Sentiment Analysis
def get_vader_sentiment(text: str) -> str:
    sentiment_score = sia.polarity_scores(text)
    if sentiment_score['compound'] >= 0.2:
        return 'positive'
    elif sentiment_score['compound'] <= -0.2:
        return 'negative'
    else:
        return 'neutral'

# Store Raw Tweets
def store_raw_tweet(connection, tweet_count, tweet) -> None:

    location = tweet.user.location if tweet.user and tweet.user.location else 'Unknown'
    tweet_id = int(tweet.id)

    # Convert created_at to MySQL-compatible format
    created_at = datetime.strptime(tweet.created_at, "%a %b %d %H:%M:%S %z %Y")
    created_at = created_at.strftime("%Y-%m-%d %H:%M:%S")

    query = """
    INSERT IGNORE INTO raw_tweets (tweet_count, tweet_id, text, created_at, location)
    VALUES (%s, %s, %s, %s, %s)
    """
    data = (tweet_count, tweet.id, tweet.text, created_at, location)
    
    cursor = connection.cursor()
    cursor.execute(query, data)
    connection.commit()
    cursor.close()


# Store Processed Tweets with Predictions
def store_processed_tweet(connection, tweet_count, tweet, sentiment, sentiment_encoded, sentiment_confidence, hashtags, keywords, brand) -> None:

    location = tweet.user.location if tweet.user and tweet.user.location else 'Unknown'
    
    tweet_id = int(tweet.id)
    sentiment_encoded = int(sentiment_encoded)

    # Convert created_at to MySQL-compatible format
    created_at = datetime.strptime(tweet.created_at, "%a %b %d %H:%M:%S %z %Y")
    created_at = created_at.strftime("%Y-%m-%d %H:%M:%S")

    query = """
    INSERT INTO processed_tweets 
    (tweet_count, tweet_id, text, created_at, sentiment, sentiment_encoded, sentiment_confidence, location, hashtags, keywords, brand)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    data = (tweet_count, tweet.id, tweet.text, created_at, sentiment, sentiment_encoded, sentiment_confidence, location, hashtags, keywords, brand)
    
    cursor = connection.cursor()
    #print("Inserting data:", data)
    cursor.execute(query, data)
    connection.commit()
    cursor.close()


# Main Twitter Pipeline
async def main() -> NoReturn:
    client.load_cookies('cookies.json')
    tweets = await client.search_tweet('#Apple', 'Latest')

    # Connect to MySQL
    connection = get_db_connection()

    try:

        # Track existing tweets to avoid duplicates
        raw_existing_ids = set()
        cursor = connection.cursor()
        cursor.execute("SELECT tweet_id FROM raw_tweets")
        raw_existing_ids = {row[0] for row in cursor.fetchall()}
        cursor.close()


        start_count = len(raw_existing_ids) + 1
        tweet_count = start_count - 1

        for tweet in tweets:
            if tweet.lang == 'en' and tweet.id not in raw_existing_ids:
                if tweet.retweeted_tweet is None and tweet.in_reply_to is None:
                    if not is_tweet_in_database(connection, tweet.id):  # Check dynamically
                        tweet_count += 1
                        print(tweet.id, tweet.text, tweet.created_at)

                        # Store raw tweet in MySQL
                        store_raw_tweet(connection, tweet_count, tweet)

                        # Extract hashtags
                        hashtags = extract_hashtags(tweet.text)

                        # Extract NER
                        brand = extract_brands(tweet.text)

                        # Preprocess Text
                        processed_text = preprocess_text(tweet.text)

                        #Extract keywords
                        keywords = extract_keywords(tweet.text)

                        # Extract BERT Features
                        bert_features = extract_bert_features_batch([processed_text])
                        bert_features_scaled = bert_scaler.transform(bert_features)

                        # Sentiment Analysis
                        sentiment = get_vader_sentiment(processed_text)
                        vader_features = [
                            sia.polarity_scores(processed_text)['pos'],
                            sia.polarity_scores(processed_text)['neg'],
                            sia.polarity_scores(processed_text)['neu'],
                            sia.polarity_scores(processed_text)['compound']
                        ]
                        X_combined = np.hstack((bert_features_scaled, np.array(vader_features).reshape(1, -1)))
                        X_combined_scaled = scaler.transform(X_combined)

                        # Predict Sentiment and Confidence Score
                        sentiment_encoded = loaded_rf_model.predict(X_combined_scaled)[0]  # Predicted class
                        sentiment_confidences = loaded_rf_model.predict_proba(X_combined_scaled)[0]  # Confidence scores
                        sentiment_confidence = float(max(sentiment_confidences))  # Maximum confidence score

                        print(f"Processed tweet: {processed_text}")
                        print(f"Predicted Sentiment: {sentiment_encoded}")
                        #print(f"Confidence Scores: {sentiment_confidences}")
                        #print(f"Sentiment Confidence: {sentiment_confidence}")


                        # Store processed tweet in MySQL
                        store_processed_tweet(connection, tweet_count, tweet, sentiment, sentiment_encoded, sentiment_confidence, hashtags, keywords, brand)

                        if tweet_count >= start_count + 19:
                            break

                        await asyncio.sleep(20)

    finally:
        if connection.is_connected():
            connection.close()
            print("MySQL connection closed.")

# Fetch sentiment data from MySQL and calculate negative percentage
def check_sentiment():
    
    engine = create_engine("mysql+mysqlconnector://root:Root%4012345@localhost/ajith")

    # Query to fetch sentiment data from the last 1 hour
    query = text("SELECT created_at, sentiment FROM processed_tweets WHERE created_at >= NOW() - INTERVAL 1 HOUR")
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error fetching data from MySQL: {e}")
        return

    negative_percentage = 0

    try:

        # Calculate negative sentiment percentage
        if not df.empty:
            negative_count = df[df['sentiment'] == 'negative'].shape[0]
            total_count = len(df)
            if total_count > 0:
                negative_percentage = (negative_count / total_count) * 100 if total_count > 0 else 0
        
        # Send alert if negative sentiment is more than 20%
        if negative_percentage >= 2:
            send_email_alert(negative_percentage)
    except Exception as e:
        logging.error(f"Database error: {e}")

    finally:

        engine.dispose()  # Close the connection
        print("MySQL connection closed.")


# Email Alerting Function
def send_email_alert(negative_percentage):
    subject = "🚨 Alert: Sudden Increase in Negative Tweets!"
    body = f"Negative sentiment has risen to {negative_percentage:.2f}% in the last hour.\nCheck the dashboard for details."
    
    sender_email = ""
    receiver_email = ""
    password = ""

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    retries = 3  # Retry up to 3 times
    for attempt in range(retries):

        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            server.quit()
            print("Email Alert Sent!")
            return
        except Exception as e:
            logging.error(f"Attempt {attempt+1} - Error sending email: {e}")
            time.sleep(2 ** attempt)

# Main pipeline to handle both tasks
async def run_pipeline():

    try:

        await main()
        # Execute the sentiment check
        check_sentiment()

    except Exception as e:
        print(f"Error in pipeline: {e}")

# Run the Script
asyncio.run(run_pipeline())

#modified reprocess pipeline with database and xgboost model 
#python -m uvicorn run_script:app --reload
# http://127.0.0.1:8000/

