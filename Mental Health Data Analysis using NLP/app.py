import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import pipeline
import asyncio

# Fix for asyncio event loop issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Set PyTorch to use a single thread to prevent crashes
torch.set_num_threads(1)

# Load the pre-trained Transformer-based emotion model
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# Emotion Mapping with Emojis
emotion_images = {
    "anger": "😡", "disgust": "🤢", "fear": "😨", "joy": "😊",
    "neutral": "😐", "sadness": "😢", "surprise": "😲",
    "love": "❤️", "optimism": "🌞", "pessimism": "☁️", 
    "excitement": "🤩", "pride": "🏆", "guilt": "😔", "shame": "🙈"
}

# Sentiment Classification Based on Emotion
sentiment_map = {
    "anger": "Negative", "disgust": "Negative", "fear": "Negative",
    "sadness": "Negative", "guilt": "Negative", "shame": "Negative",
    "joy": "Positive", "love": "Positive", "optimism": "Positive",
    "excitement": "Positive", "pride": "Positive",
    "neutral": "Neutral", "surprise": "Neutral", "pessimism": "Neutral"
}

# Emotion Explanation & Coping Strategies (Expanded)
emotion_strategies = {
    "anger": ("Anger is a strong feeling of annoyance or hostility.", [
        "Practice deep breathing.",
        "Go for a walk to cool down.",
        "Express yourself through writing."
    ]),
    "sadness": ("Sadness is a natural response to loss or disappointment.", [
        "Talk to a friend or journal your feelings.",
        "Engage in enjoyable activities.",
        "Practice mindfulness and gratitude."
    ]),
    "fear": ("Fear is a response to perceived danger, real or imagined.", [
        "Try slow breathing techniques.",
        "Challenge irrational thoughts.",
        "Gradual exposure to fears can help."
    ]),
    "joy": ("Joy is a feeling of great pleasure and happiness.", [
        "Share your happiness with others.",
        "Express gratitude.",
        "Engage in creative activities."
    ]),
    "neutral": ("A neutral state often means emotional stability.", [
        "Reflect on your current state.",
        "Use this moment for self-improvement.",
        "Engage in light activities to boost your mood."
    ]),
    "surprise": ("Surprise is an emotional reaction to something unexpected.", [
        "Embrace spontaneity.",
        "Use it as a learning opportunity.",
        "Channel the energy into something positive."
    ]),
    "love": ("Love is a deep feeling of affection and care.", [
        "Spend quality time with loved ones.",
        "Practice self-love and compassion.",
        "Express your appreciation openly."
    ]),
    "disgust": ("Disgust is a strong disapproval or revulsion reaction.", [
        "Try to understand the source of discomfort.",
        "Talk about your feelings calmly.",
        "Avoid distressing situations when possible."
    ]),
    "pessimism": ("Pessimism is a tendency to expect the worst.", [
        "Reframe negative thoughts into positive ones.",
        "Focus on small daily achievements.",
        "Limit exposure to negativity."
    ]),
    "excitement": ("Excitement is a state of enthusiastic anticipation.", [
        "Channel your energy into something productive.",
        "Share your excitement with others.",
        "Stay grounded with deep breathing."
    ]),
    "pride": ("Pride is a feeling of deep satisfaction from achievements.", [
        "Celebrate your success mindfully.",
        "Recognize the efforts of others.",
        "Stay humble and motivated."
    ]),
    "guilt": ("Guilt is the feeling of remorse over past actions.", [
        "Acknowledge the mistake and make amends.",
        "Forgive yourself and learn from the experience.",
        "Talk to someone you trust."
    ]),
    "shame": ("Shame is a painful feeling of humiliation or distress.", [
        "Challenge negative self-beliefs.",
        "Speak with a supportive person.",
        "Practice self-compassion."
    ])
}

# ---------------- UI ----------------
st.title("🔍 Mental Health Emotion Analysis Using NLP")
st.markdown("Analyze text emotions or upload a CSV for batch analysis.")

mode = st.radio("Choose Input Mode:", ["Single Text Input", "CSV Upload"], index=None)

# ---------------- Text Input Mode ----------------
if mode == "Single Text Input":
    user_input = st.text_area("Enter text to analyze:", help="Type a sentence to detect its emotion and sentiment.")
    
    if st.button("Analyze Emotion"):
        if user_input.strip():
            emotion_scores = emotion_model(user_input)[0]
            sorted_emotions = sorted(emotion_scores, key=lambda x: x['score'], reverse=True)
            predicted_emotion = sorted_emotions[0]['label']
            detected_emoji = emotion_images.get(predicted_emotion, "❓")
            sentiment = sentiment_map.get(predicted_emotion, "Unknown")

            st.subheader(f"Detected Emotion: {predicted_emotion} {detected_emoji}")
            st.write(f"**Sentiment Analysis:** {sentiment}")

            # Explanation & Coping Strategy or Fallback
            if predicted_emotion in emotion_strategies:
                explanation, strategies = emotion_strategies[predicted_emotion]
                st.markdown(f"**📖 Explanation:** {explanation}")
                st.markdown("**💡 Coping Strategies:**")
                for strategy in strategies:
                    st.write(f"✔️ {strategy}")
            else:
                st.markdown("🧠 *No predefined strategies available for this emotion yet. Try journaling or speaking with someone you trust.*")

            # Confidence Scores
            with st.expander("🔎 Analyzed Model Confidence Scores"):
                for item in sorted_emotions:
                    st.write(f"{item['label']}: {item['score']:.4f}")

            # Visualization
            labels = [item['label'] for item in sorted_emotions]
            scores = [item['score'] for item in sorted_emotions]

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = sns.barplot(y=labels, x=scores, hue=labels, palette='coolwarm', legend=False, ax=ax)

            for i, (bar, score) in enumerate(zip(bars.patches, scores)):
                ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, f"{score:.2f}  ", color='black', va="center", fontsize=10)
                if labels[i] == predicted_emotion:
                    ax.text(score + 0.08, bar.get_y() + bar.get_height()/2, detected_emoji, fontsize=14, va="center")

            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Emotion Category")
            ax.set_title("Emotion Probability Distribution")
            ax.set_xlim(0, 1)
            st.pyplot(fig)
        else:
            st.warning("⚠️ Please enter some text to analyze.")

# ---------------- CSV Upload Mode ----------------  
elif mode == "CSV Upload":
    st.subheader(":open_file_folder: Upload CSV for Bulk Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file with at least one text column", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📊 Preview of Uploaded Data:")
        st.dataframe(df.head())

        # Automatically detect the text column (based on longest average string length)
        text_column = None
        max_avg_length = 0
        for col in df.columns:
            if df[col].dtype == object:
                avg_len = df[col].dropna().astype(str).apply(len).mean()
                if avg_len > max_avg_length:
                    max_avg_length = avg_len
                    text_column = col

        if text_column:
            

            def analyze_row(text):
                emotions = emotion_model(str(text))[0]
                top_emotion = sorted(emotions, key=lambda x: x['score'], reverse=True)[0]['label']
                sentiment = sentiment_map.get(top_emotion, "Unknown")
                return top_emotion, sentiment

            df[['Emotion_Analysis', 'Sentiment_Analysis']] = df[text_column].apply(lambda x: pd.Series(analyze_row(x)))

            st.subheader(":page_facing_up: Analysis Results Table")
            st.dataframe(df)

            st.subheader(":bar_chart: Emotion & Sentiment Distribution")

            fig1, ax1 = plt.subplots()
            sns.countplot(data=df, x='Emotion_Analysis', order=df['Emotion_Analysis'].value_counts().index, palette='Set3', ax=ax1)
            ax1.set_title('Emotion Distribution')
            ax1.set_xlabel('Emotion')
            ax1.set_ylabel('Count')
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            sns.countplot(data=df, x='Sentiment_Analysis', order=df['Sentiment_Analysis'].value_counts().index, palette='Set2', ax=ax2)
            ax2.set_title('Sentiment Distribution')
            ax2.set_xlabel('Sentiment')
            ax2.set_ylabel('Count')
            st.pyplot(fig2)
        else:
            st.error("❌ Could not find a suitable Mental Health column in the uploaded file.")

else:
    st.info("ℹ️ Please select an input mode to begin.")