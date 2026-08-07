import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="✈️ Airline Sentiment Analyzer",
    page_icon="✈️",
    layout="wide"
)

# ============================================================================
# LOAD MODELS WITH ERROR HANDLING
# ============================================================================

@st.cache_resource
def load_models():
    """Load trained models with fallback"""
    try:
        model = joblib.load('best_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        encoder = joblib.load('encoder.pkl')
        return model, vectorizer, encoder
    except Exception as e:
        st.warning(f"⚠️ Model loading error: {str(e)[:100]}...")
        return None, None, None

# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================================================
# PREDICTION
# ============================================================================

def predict_sentiment(text, model, vectorizer, encoder):
    if not text or text.strip() == "":
        return "neutral", 50.0
    
    cleaned = clean_text(text)
    X_text = vectorizer.transform([cleaned])
    pred = model.predict(X_text)
    
    # Get confidence
    try:
        proba = model.predict_proba(X_text)
        confidence = np.max(proba) * 100
    except:
        confidence = 75.0
    
    sentiment = encoder.inverse_transform(pred)[0]
    return sentiment, confidence

# ============================================================================
# SAMPLE TWEETS
# ============================================================================

SAMPLE_TWEETS = [
    ("I absolutely love Virgin America! Best airline ever! 🥰", "positive"),
    ("Worst airline ever. Flight delayed 5 hours. Terrible service.", "negative"),
    ("The flight was okay, nothing special.", "neutral"),
    ("Great service, very friendly staff!", "positive"),
]

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("✈️ Airline Sentiment Analyzer")
    st.markdown("*Real-time sentiment analysis for airline tweets*")
    st.markdown("---")
    
    model, vectorizer, encoder = load_models()
    
    # Show warning if models not loaded
    if model is None:
        st.error("""
        ⚠️ **Model not loaded!**
        
        Please make sure the model files are uploaded to the repository.
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.markdown("""
        This app analyzes sentiment of airline tweets.
        
        **Model Details:**
        - **Model:** SGD Classifier
        - **Accuracy:** 78.01%
        - **Training Data:** 11,602 tweets
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter a Tweet")
        tweet = st.text_area(
            "Type or paste a tweet about an airline:",
            height=100,
            placeholder="e.g., I absolutely love flying with Virgin America!"
        )
        
        analyze_button = st.button("🚀 Analyze Sentiment", type="primary")
        
        if analyze_button and tweet:
            sentiment, confidence = predict_sentiment(tweet, model, vectorizer, encoder)
            
            st.markdown("---")
            st.subheader("🎯 Prediction Result")
            
            if sentiment == "positive":
                st.success(f"✅ **Sentiment: POSITIVE** ({confidence:.1f}% confidence)")
            elif sentiment == "negative":
                st.error(f"❌ **Sentiment: NEGATIVE** ({confidence:.1f}% confidence)")
            else:
                st.warning(f"⚠️ **Sentiment: NEUTRAL** ({confidence:.1f}% confidence)")
                
        elif analyze_button and not tweet:
            st.warning("⚠️ Please enter a tweet first!")
    
    with col2:
        st.subheader("🔬 Quick Samples")
        for i, (text, _) in enumerate(SAMPLE_TWEETS):
            if st.button(f"Sample {i+1}", key=f"sample_{i}"):
                sentiment, confidence = predict_sentiment(text, model, vectorizer, encoder)
                if sentiment == "positive":
                    st.success(f"✅ {sentiment.upper()}")
                elif sentiment == "negative":
                    st.error(f"❌ {sentiment.upper()}")
                else:
                    st.warning(f"⚠️ {sentiment.upper()}")
                st.caption(f"📝 {text[:50]}...")

if __name__ == "__main__":
    main()
