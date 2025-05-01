from textblob import TextBlob
import pandas as pd

def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiments(filepath):
    df = pd.read_csv(filepath)
    df["Sentiment"] = df["cleaned_text"].apply(get_sentiment)
    df.to_csv("../data/sentiment_analysis.csv", index=False)
    return df

if __name__ == "__main__":
    df = analyze_sentiments("../data/mental_health_tweets.csv")
    print(df.head())
