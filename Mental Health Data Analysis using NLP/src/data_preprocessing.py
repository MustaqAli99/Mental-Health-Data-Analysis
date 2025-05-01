import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@w+|\#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def preprocess_dataset(filepath):
    df = pd.read_csv(filepath)
    df.dropna(subset=["text"], inplace=True)
    df["cleaned_text"] = df["text"].apply(clean_text)
    return df

if __name__ == "__main__":
    df = preprocess_dataset("../data/mental_health_tweets.csv")
    df.to_csv("../data/processed_tweets.csv", index=False)
