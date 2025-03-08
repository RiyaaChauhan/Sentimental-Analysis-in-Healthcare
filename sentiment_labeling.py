from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon
# nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Convert continuous sentiment scores into discrete classes (positive, negative, neutral)
def categorize_sentiment(score):
    if score > 0:
        return 1  # Positive sentiment
    elif score < 0:
        return -1  # Negative sentiment
    else:
        return 0  # Neutral sentiment

# Label the sentiment based on VADER scores
def apply_sentiment_labeling(data):
    data['sentiment'] = data['review'].apply(lambda x: sia.polarity_scores(x)['compound'])
    data['sentiment_category'] = data['sentiment'].apply(categorize_sentiment)
    return data
