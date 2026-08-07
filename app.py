import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import pickle

st.set_page_config(
    page_title="✈️ Airline Sentiment Analyzer",
    page_icon="✈️",
    layout="wide"
)

@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        encoder = joblib.load('encoder.pkl')
        return model, vectorizer, encoder
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(text, model, vectorizer, encoder):
    if not text or text.strip() == "":
        return "neutral", 50.0
    
    cleaned = clean_text(text)
    X_text = vectorizer.transform([cleaned])
    pred = model.predict(X_text)[0]
    sentiment = encoder.inverse_transform([pred])[0]
    
    try:
        proba = model.predict_proba(X_text)
        confidence = np.max(proba) * 100
    except:
        confidence = 75.0
    
    return sentiment, confidence

def main():
    st.title("✈️ Airline Sentiment Analyzer")
    st.markdown("*Real-time sentiment analysis for airline tweets*")
    
    model, vectorizer, encoder = load_models()
    
    if model is None:
        st.warning("""
        ⚠️ **Model files not loaded!**
        
        Please ensure these files are in the app directory:
        - `best_model.pkl`
        - `vectorizer.pkl`
        - `encoder.pkl`
        """)
        return
    
    with st.sidebar:
        st.header("📊 Model Info")
        st.metric("Model", "SGD Classifier")
        st.metric("Accuracy", "78.01%")
        st.metric("F1-Macro", "71.66%")
        
        st.markdown("---")
        st.markdown("### 📝 Example Tweets")
        st.code("I love this airline!", language="text")
        st.code("Worst flight ever!", language="text")
        st.code("It was okay.", language="text")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter a Tweet")
        
        tweet = st.text_area(
            "Type or paste a tweet about an airline:",
            height=100,
            placeholder="e.g., I absolutely love flying with Virgin America!"
        )
        
        if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
            if tweet and tweet.strip():
                sentiment, confidence = predict_sentiment(tweet, model, vectorizer, encoder)
                
                st.markdown("---")
                st.subheader("🎯 Prediction Result")
                
                if sentiment == "positive":
                    st.success(f"✅ **POSITIVE** ({confidence:.1f}% confidence)")
                elif sentiment == "negative":
                    st.error(f"❌ **NEGATIVE** ({confidence:.1f}% confidence)")
                else:
                    st.warning(f"⚠️ **NEUTRAL** ({confidence:.1f}% confidence)")
            else:
                st.warning("⚠️ Please enter a tweet first!")
    
    with col2:
        st.subheader("🔬 Try These Examples")
        
        examples = {
            "❤️ Positive": "I absolutely love Virgin America! Best airline ever!",
            "💔 Negative": "Worst airline ever. Flight delayed 5 hours.",
            "😐 Neutral": "The flight was okay, nothing special."
        }
        
        for label, text in examples.items():
            if st.button(label, key=label, use_container_width=True):
                sentiment, confidence = predict_sentiment(text, model, vectorizer, encoder)
                emoji = "✅" if sentiment == "positive" else "❌" if sentiment == "negative" else "⚠️"
                st.info(f"{emoji} {sentiment.upper()} ({confidence:.1f}%)")

if __name__ == "__main__":
    main()
