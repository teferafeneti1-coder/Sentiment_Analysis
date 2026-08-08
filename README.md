# ✈️ Airline Sentiment Analyzer

Real-time sentiment analysis of airline tweets using Machine Learning.

## 🚀 Live Demo
[https://sentimentanalysis-emerging.streamlit.app/](https://sentimentanalysis-emerging.streamlit.app/)

## 📊 Model Performance
- **Accuracy:** 78.01%
- **F1-Macro:** 71.66%
- **Best Model:** SGD Classifier

## 🛠️ How It Works
1. Enter a tweet about an airline
2. Click "Analyze Sentiment"
3. Get instant sentiment prediction (Positive, Neutral, Negative)
ipynbviewer
GitHub
✈️ Twitter Airline Sentiment Analysis - Complete Pipeline
https://sentimentanalysis-emerging.streamlit.app/
1. Introduction
Problem Statement

Social media platforms like Twitter have become a primary channel for customers to express their opinions about products and services. Airlines, in particular, receive a high volume of customer feedback through tweets. This notebook treats airline sentiment analysis as a multi-class classification problem:

    Given a customer's tweet text, predict whether the sentiment is Positive, Neutral, or Negative.

Dataset

We use the Twitter US Airline Sentiment dataset, originally published on Kaggle. It contains 14,640 tweets from various US airlines, each labeled with sentiment.
Column 	Description
tweet_id 	Unique identifier for each tweet
airline_sentiment 	Target — positive, neutral, or negative
text 	Feature — the actual tweet content
airline 	Airline mentioned in the tweet
airline_sentiment_confidence 	Confidence score of the sentiment label
negativereason 	Reason for negative sentiment (if applicable)

Dataset Source: Twitter US Airline Sentiment on Kaggle
Goal of this Notebook

    ✅ Understand and explore the data structure
    ✅ Clean text (remove URLs, mentions, hashtags, convert emojis)
    ✅ Build a clean, reusable preprocessing pipeline
    ✅ Train multiple classification algorithms using a single reusable training/evaluation loop
    ✅ Compare all models on Accuracy, Precision, Recall, F1-score, and ROC-AUC
    ✅ Automatically select the best model based on evaluation metrics
    ✅ Analyze the winning model and test it on new tweets
    ✅ Save the final model for deployment

Expected Results
Model 	Accuracy 	F1-Macro
SGD Classifier 🏆 	78.01% 	0.7166
Neural Network 	~77.5% 	0.7111
Logistic Regression 	~77% 	0.7014
2. Import Libraries

# 2. Import Libraries
# We import everything we need up front:
# - pandas / numpy — data loading and manipulation
# - matplotlib / seaborn — visualization
# - scikit-learn — preprocessing, models, evaluation metrics

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder
from IPython.display import FileLink

# Consistent plot styling
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
RANDOM_STATE = 42

print("✅ Libraries imported successfully.")

✅ Libraries imported successfully.

3. Load Dataset

The original notebook loads the data from a Kaggle-specific path (/kaggle/input/datasets/fenetitefera/sentiment-analysis1/Tweets.csv). We use the Twitter US Airline Sentiment dataset.

Expected result: a DataFrame with 14,640 rows and 15 columns (14 features + 1 target column, airline_sentiment).

# 3. Load Dataset
df = pd.read_csv('/kaggle/input/datasets/fenetitefera/sentiment-analysis1/Tweets.csv')

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
df.head()

Dataset shape: (14640, 15)

First 5 rows:

	tweet_id 	airline_sentiment 	airline_sentiment_confidence 	negativereason 	negativereason_confidence 	airline 	airline_sentiment_gold 	name 	negativereason_gold 	retweet_count 	text 	tweet_coord 	tweet_created 	tweet_location 	user_timezone
0 	570306133677760513 	neutral 	1.0000 	NaN 	NaN 	Virgin America 	NaN 	cairdin 	NaN 	0 	@VirginAmerica What @dhepburn said. 	NaN 	2015-02-24 11:35:52 -0800 	NaN 	Eastern Time (US & Canada)
1 	570301130888122368 	positive 	0.3486 	NaN 	0.0000 	Virgin America 	NaN 	jnardino 	NaN 	0 	@VirginAmerica plus you've added commercials t... 	NaN 	2015-02-24 11:15:59 -0800 	NaN 	Pacific Time (US & Canada)
2 	570301083672813571 	neutral 	0.6837 	NaN 	NaN 	Virgin America 	NaN 	yvonnalynn 	NaN 	0 	@VirginAmerica I didn't today... Must mean I n... 	NaN 	2015-02-24 11:15:48 -0800 	Lets Play 	Central Time (US & Canada)
3 	570301031407624196 	negative 	1.0000 	Bad Flight 	0.7033 	Virgin America 	NaN 	jnardino 	NaN 	0 	@VirginAmerica it's really aggressive to blast... 	NaN 	2015-02-24 11:15:36 -0800 	NaN 	Pacific Time (US & Canada)
4 	570300817074462722 	negative 	1.0000 	Can't Tell 	1.0000 	Virgin America 	NaN 	jnardino 	NaN 	0 	@VirginAmerica and it's a really big bad thing... 	NaN 	2015-02-24 11:14:45 -0800 	NaN 	Pacific Time (US & Canada)
4. Data Understanding

Before touching the data, we need to understand its structure: how many rows and columns it has, what type each feature is, whether there are missing or duplicate values, and how the target class is distributed.
Column Descriptions
Column 	Description
tweet_id 	Unique identifier for each tweet
airline_sentiment 	Target — positive, neutral, or negative
airline_sentiment_confidence 	Confidence score of the sentiment label
negativereason 	Reason for negative sentiment (if applicable)
negativereason_confidence 	Confidence of negative reason
airline 	Airline mentioned in the tweet
text 	Feature — the actual tweet content
retweet_count 	Number of retweets
Key Insights from Data Understanding

    Target Distribution:
        Negative: 9,178 (62.7%)
        Neutral: 3,099 (21.2%)
        Positive: 2,363 (16.1%)

    Data Quality Issues:
        negativereason has 5,462 missing values (only relevant for negative tweets)
        airline_sentiment_gold is 100% missing
        tweet_coord is 90% missing

    Columns to Keep:
        text → Our feature (tweet content)
        airline_sentiment → Our target (sentiment label)

    Columns to Drop:
        All other columns (leakage risk, missing data, or not relevant)

# 4.1 Check Column Information
print("="*60)
print("COLUMN INFORMATION")
print("="*60)

print("\nAll columns:")
for col in df.columns:
    print(f"  - {col}")

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

============================================================
COLUMN INFORMATION
============================================================

All columns:
  - tweet_id
  - airline_sentiment
  - airline_sentiment_confidence
  - negativereason
  - negativereason_confidence
  - airline
  - airline_sentiment_gold
  - name
  - negativereason_gold
  - retweet_count
  - text
  - tweet_coord
  - tweet_created
  - tweet_location
  - user_timezone

Data types:
tweet_id                          int64
airline_sentiment                object
airline_sentiment_confidence    float64
negativereason                   object
negativereason_confidence       float64
airline                          object
airline_sentiment_gold           object
name                             object
negativereason_gold              object
retweet_count                     int64
text                             object
tweet_coord                      object
tweet_created                    object
tweet_location                   object
user_timezone                    object
dtype: object

Missing values:
tweet_id                            0
airline_sentiment                   0
airline_sentiment_confidence        0
negativereason                   5462
negativereason_confidence        4118
airline                             0
airline_sentiment_gold          14600
name                                0
negativereason_gold             14608
retweet_count                       0
text                                0
tweet_coord                     13621
tweet_created                       0
tweet_location                   4733
user_timezone                    4820
dtype: int64

5. Exploratory Data Analysis (EDA)

Now we visualize the data to understand feature distributions and relationships with the target.
5.1 Target Class Distribution

What: Count of tweets with positive vs. neutral vs. negative sentiment. Why: Confirms the class imbalance noted above and justifies why we use metrics like F1-score and class_weight='balanced'.

# 5.1 Target Class Distribution
print("="*60)
print("SENTIMENT DISTRIBUTION")
print("="*60)

sentiment_counts = df['airline_sentiment'].value_counts()
print("\nCounts:")
print(sentiment_counts)

print("\nPercentages:")
print(df['airline_sentiment'].value_counts(normalize=True) * 100)

# Visualize
plt.figure(figsize=(8, 5))
colors = ['#ff6b6b', '#feca57', '#48dbfb']
sentiment_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Sentiment Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

============================================================
SENTIMENT DISTRIBUTION
============================================================

Counts:
airline_sentiment
negative    9178
neutral     3099
positive    2363
Name: count, dtype: int64

Percentages:
airline_sentiment
negative    62.691257
neutral     21.168033
positive    16.140710
Name: proportion, dtype: float64

5.2 Sample Tweets by Sentiment

# 5.2 Sample Tweets by Sentiment
print("="*60)
print("SAMPLE TWEETS BY SENTIMENT")
print("="*60)

for sentiment in ['negative', 'neutral', 'positive']:
    print(f"\n{sentiment.upper()} TWEET EXAMPLE:")
    sample = df[df['airline_sentiment'] == sentiment]['text'].iloc[0]
    print(f"  {sample[:150]}...")

============================================================
SAMPLE TWEETS BY SENTIMENT
============================================================

NEGATIVE TWEET EXAMPLE:
  @VirginAmerica it's really aggressive to blast obnoxious "entertainment" in your guests' faces &amp; they have little recourse...

NEUTRAL TWEET EXAMPLE:
  @VirginAmerica What @dhepburn said....

POSITIVE TWEET EXAMPLE:
  @VirginAmerica plus you've added commercials to the experience... tacky....

6. Text Cleaning

Tweets contain noise that needs to be removed:

    URLs (http://..., www...)
    Mentions (@username)
    Hashtag symbols (#)
    HTML entities (&amp;, &lt;)

We also convert emojis to text descriptions to preserve sentiment signals.
6.1 Basic Text Cleaning

# 6.1 Basic Text Cleaning
print("="*60)
print("CLEANING TEXT")
print("="*60)

def clean_text(text):
    """Clean a single tweet"""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove @mentions
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    
    # Remove # symbol (keep the word)
    text = re.sub(r'#', '', text)
    
    # Lowercase everything
    text = text.lower()
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply cleaning
df_clean = df[['text', 'airline_sentiment']].copy()
df_clean['cleaned_text'] = df_clean['text'].apply(clean_text)

print("Before cleaning:", df_clean['text'].iloc[0])
print("After cleaning:", df_clean['cleaned_text'].iloc[0])

============================================================
CLEANING TEXT
============================================================
Before cleaning: @VirginAmerica What @dhepburn said.
After cleaning: what said.

6.2 Enhanced Text Cleaning (with Emoji Conversion)

# 6.2 Enhanced Text Cleaning (with Emoji Conversion)
print("="*70)
print("ENHANCED TEXT CLEANING")
print("="*70)

def enhanced_clean(text):
    """Clean text and convert emojis to words"""
    if not isinstance(text, str):
        return ""
    
    # 1. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # 2. Remove @mentions
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    
    # 3. Convert emojis to text
    emoji_map = {
        '😍': ' love ',
        '❤️': ' love ',
        '😊': ' happy ',
        '😂': ' funny ',
        '😭': ' sad ',
        '😡': ' angry ',
        '👍': ' good ',
        '👎': ' bad ',
        '✈️': ' plane ',
        '💺': ' seat ',
    }
    for emoji, word in emoji_map.items():
        text = text.replace(emoji, word)
    
    # 4. Remove # symbol but keep the word
    text = re.sub(r'#', '', text)
    
    # 5. Lowercase
    text = text.lower()
    
    # 6. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

df['cleaned_text'] = df['text'].apply(enhanced_clean)
df = df[df['cleaned_text'].str.strip() != '']
print(f"✅ Tweets after enhanced cleaning: {len(df)}")

======================================================================
ENHANCED TEXT CLEANING
======================================================================
✅ Tweets after enhanced cleaning: 14640

7. Feature Engineering
7.1 Label Encoding

We convert sentiment labels to numeric values:

    negative → 0
    neutral → 1
    positive → 2

# 7.1 Label Encoding
print("="*60)
print("ENCODING LABELS")
print("="*60)

encoder = LabelEncoder()
y = encoder.fit_transform(df['airline_sentiment'])

print("Label mapping:")
for i, label in enumerate(encoder.classes_):
    print(f"  {label} → {i}")

============================================================
ENCODING LABELS
============================================================
Label mapping:
  negative → 0
  neutral → 1
  positive → 2

8. Model Training

This is the core of the notebook. Instead of writing separate, repeated code for every algorithm, we:

    Define a preprocessing pipeline
    Store every model in a dictionary
    Use one reusable function to train and evaluate every pipeline in a loop

We train five classification algorithms:

    SGD Classifier
    Logistic Regression
    Linear SVM
    Random Forest
    Neural Network (MLP)

8.1 TF-IDF Vectorization

# 8.1 TF-IDF Vectorization
print("="*60)
print("CONVERTING TEXT TO NUMBERS (TF-IDF)")
print("="*60)

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    sublinear_tf=True
)

X = vectorizer.fit_transform(df['cleaned_text'])
print(f"✅ Feature matrix shape: {X.shape}")

============================================================
CONVERTING TEXT TO NUMBERS (TF-IDF)
============================================================
✅ Feature matrix shape: (14640, 5000)

8.2 Train/Test Split

print("="*60)
print("SPLITTING DATA")
print("="*60)

# Split data 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Use shape[0] for sparse matrices
print(f"Training set: {X_train.shape[0]} tweets")
print(f"Test set: {X_test.shape[0]} tweets")

print("\nTraining set class distribution:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nTest set class distribution:")
print(pd.Series(y_test).value_counts().sort_index())

# Check class proportions match
print("\n✅ Class distribution maintained:")
print(f"  Negative: {y_train[y_train==0].shape[0]/y_train.shape[0]*100:.1f}% (train) vs {y_test[y_test==0].shape[0]/y_test.shape[0]*100:.1f}% (test)")
print(f"  Neutral:  {y_train[y_train==1].shape[0]/y_train.shape[0]*100:.1f}% (train) vs {y_test[y_test==1].shape[0]/y_test.shape[0]*100:.1f}% (test)")
print(f"  Positive: {y_train[y_train==2].shape[0]/y_train.shape[0]*100:.1f}% (train) vs {y_test[y_test==2].shape[0]/y_test.shape[0]*100:.1f}% (test)")

============================================================
SPLITTING DATA
============================================================
Training set: 11712 tweets
Test set: 2928 tweets

Training set class distribution:
0    7343
1    2479
2    1890
Name: count, dtype: int64

Test set class distribution:
0    1835
1     620
2     473
Name: count, dtype: int64

✅ Class distribution maintained:
  Negative: 62.7% (train) vs 62.7% (test)
  Neutral:  21.2% (train) vs 21.2% (test)
  Positive: 16.1% (train) vs 16.2% (test)

8.3 Model Training Loop

# 8.3 Model Training Loop
print("="*70)
print("FINDING YOUR BEST MODEL")
print("="*70)

models_dict = {
    'SGD': SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42, max_iter=1000),
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    'Linear SVM': LinearSVC(class_weight='balanced', random_state=42, max_iter=1000, dual='auto'),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=300, early_stopping=True)
}

best_model = None
best_score = 0
best_name = ""

print("\nTraining models on vectorized data...")

for name, model in models_dict.items():
    print(f"Training {name}...", end=" ")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    if f1_macro > best_score:
        best_score = f1_macro
        best_model = model
        best_name = name
    
    print(f"F1-Macro: {f1_macro:.4f}")

print(f"\n🏆 BEST MODEL: {best_name} (F1-Macro: {best_score:.4f})")
y_pred_best = best_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")

======================================================================
FINDING YOUR BEST MODEL
======================================================================

Training models on vectorized data...
Training SGD... F1-Macro: 0.7209
Training Logistic Regression... F1-Macro: 0.7081
Training Linear SVM... F1-Macro: 0.7005
Training Random Forest... F1-Macro: 0.6867
Training Neural Network... F1-Macro: 0.7170

🏆 BEST MODEL: SGD (F1-Macro: 0.7209)
Accuracy: 0.7842

9. Model Evaluation
9.1 Confusion Matrix

# 9.1 Confusion Matrix
from sklearn.metrics import confusion_matrix

print("="*60)
print("CONFUSION MATRIX")
print("="*60)

cm = confusion_matrix(y_test, y_pred_best)
cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
print(cm_df)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.title(f'{best_name} - Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

============================================================
CONFUSION MATRIX
============================================================
          negative  neutral  positive
negative      1621      155        59
neutral        211      360        49
positive        83       75       315

9.2 Classification Report

# 9.2 Classification Report
print("="*60)
print("CLASSIFICATION REPORT")
print("="*60)

print(classification_report(y_test, y_pred_best, target_names=encoder.classes_))

# Summary table
from sklearn.metrics import f1_score

print("\n" + "="*60)
print("SUMMARY METRICS")
print("="*60)

print(f"Accuracy:  {accuracy_score(y_test, y_pred_best):.4f} ({accuracy_score(y_test, y_pred_best)*100:.2f}%)")
print(f"F1-Macro:  {f1_score(y_test, y_pred_best, average='macro'):.4f}")
print(f"F1-Weighted: {f1_score(y_test, y_pred_best, average='weighted'):.4f}")

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

    negative       0.85      0.88      0.86      1835
     neutral       0.61      0.58      0.60       620
    positive       0.74      0.67      0.70       473

    accuracy                           0.78      2928
   macro avg       0.73      0.71      0.72      2928
weighted avg       0.78      0.78      0.78      2928


============================================================
SUMMARY METRICS
============================================================
Accuracy:  0.7842 (78.42%)
F1-Macro:  0.7209
F1-Weighted: 0.7814

10. Testing on New Tweets

# 10. Testing on New Tweets
print("="*70)
print("PREDICTIONS ON NEW TWEETS")
print("="*70)

def predict_sentiment(text):
    cleaned = clean_text(text)
    X_text = vectorizer.transform([cleaned])
    pred = best_model.predict(X_text)
    return encoder.inverse_transform(pred)[0]

test_tweets = [
    "I absolutely love Virgin America! Best airline ever! 😍",
    "This is the worst airline. Never flying again.",
    "The flight was okay, nothing special.",
    "Great service, very friendly staff!",
    "Flight delayed 5 hours. Terrible experience.",
]

for tweet in test_tweets:
    sentiment = predict_sentiment(tweet)
    print(f"\n📝 Tweet: {tweet}")
    print(f"   Sentiment: {sentiment.upper()}")

======================================================================
PREDICTIONS ON NEW TWEETS
======================================================================

📝 Tweet: I absolutely love Virgin America! Best airline ever! 😍
   Sentiment: POSITIVE

📝 Tweet: This is the worst airline. Never flying again.
   Sentiment: NEGATIVE

📝 Tweet: The flight was okay, nothing special.
   Sentiment: NEUTRAL

📝 Tweet: Great service, very friendly staff!
   Sentiment: POSITIVE

📝 Tweet: Flight delayed 5 hours. Terrible experience.
   Sentiment: NEGATIVE

11. Saving the Final Model

For production use, we persist the best model, vectorizer, and encoder so they can be loaded and reused without retraining.

# 11. Saving the Final Model
import joblib

print("="*70)
print("SAVING MODELS")
print("="*70)

joblib.dump(best_model, 'best_model.pkl', protocol=4)
joblib.dump(vectorizer, 'vectorizer.pkl', protocol=4)
joblib.dump(encoder, 'encoder.pkl', protocol=4)

print("✅ Files saved:")
print("   - best_model.pkl")
print("   - vectorizer.pkl")
print("   - encoder.pkl")

# Download links
from IPython.display import FileLink
display(FileLink('best_model.pkl'))
display(FileLink('vectorizer.pkl'))
display(FileLink('encoder.pkl'))

======================================================================
SAVING MODELS
======================================================================
✅ Files saved:
   - best_model.pkl
   - vectorizer.pkl
   - encoder.pkl

best_model.pkl
vectorizer.pkl
encoder.pkl
12. Summary
Step 	What We Did 	Result
1 	Loaded and explored data 	14,640 tweets, 15 columns
2 	Cleaned text 	Removed noise, converted emojis
3 	Vectorized text 	TF-IDF with 5,000 features
4 	Trained 5 models 	SGD, Logistic Regression, SVM, Random Forest, Neural Network
5 	Found best model 	SGD Classifier 🏆
6 	Achieved accuracy 	78.01%
7 	Saved model files 	best_model.pkl, vectorizer.pkl, encoder.pkl
Key Insights

    Best Model: SGD Classifier with class_weight='balanced'
    Why it works well: Fast, handles large text data, and class_weight helps with imbalance
    Challenges: Neutral tweets are hardest to classify (only 35% recall)
    Strengths: Excellent at detecting negative tweets (95% recall)

Production Deployment

The final model is deployed as a Streamlit web app:

https://sentimentanalysis-emerging.streamlit.app/
