import pandas as pd
from transformers import pipeline

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

def detect_emotion(text):
    result = emotion_classifier(text)[0][0]
    return result["label"]

def analyze_emotions(filepath):
    df = pd.read_csv(filepath)
    df["Emotion"] = df["cleaned_text"].apply(detect_emotion)
    df.to_csv("../data/emotion_analysis.csv", index=False)
    return df

if __name__ == "__main__":
    df = analyze_emotions("../data/processed_tweets.csv")
    print(df.head())
