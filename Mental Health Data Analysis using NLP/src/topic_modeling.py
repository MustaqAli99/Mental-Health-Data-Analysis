import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def extract_topics(filepath, num_topics=5):
    df = pd.read_csv(filepath)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    X = vectorizer.fit_transform(df["cleaned_text"])

    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_model.fit(X)

    words = vectorizer.get_feature_names_out()
    topics = []
    for i, topic in enumerate(nmf_model.components_):
        topic_words = [words[i] for i in topic.argsort()[-10:]]
        topics.append(" ".join(topic_words))

    return topics

if __name__ == "__main__":
    topics = extract_topics("../data/processed_tweets.csv")
    print("Extracted Topics:", topics)
