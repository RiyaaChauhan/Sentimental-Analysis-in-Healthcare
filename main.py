import pandas as pd
from preprocessing import load_and_preprocess_data
from sentiment_labeling import apply_sentiment_labeling
from TF_IDF import apply_tfidf
from Model import train_model
from evaluation import evaluate_model
from tqdm import tqdm
import nltk

# # Download NLTK resources if not already downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

# File paths for the train and test dataset
train_file = "drugsComTrain_raw.tsv"
test_file = "drugsComTest_raw.tsv"

# Step 1: Load and preprocess the data
print("Step 1: Loading and preprocessing the data...")
for _ in tqdm(range(1), desc="Loading & Preprocessing"):
    data = load_and_preprocess_data(train_file, test_file)

# Step 2: Apply sentiment labeling
print("\nStep 2: Applying sentiment labeling...")
for _ in tqdm(range(1), desc="Sentiment Labeling"):
    data = apply_sentiment_labeling(data)

# Step 3: Apply TF-IDF vectorization
print("\nStep 3: Applying TF-IDF vectorization...")
for _ in tqdm(range(1), desc="TF-IDF Vectorization"):
    X, tfidf_vectorizer = apply_tfidf(data)

# Step 4: Train the sentiment classification model
print("\nStep 4: Training the sentiment classification model...")
y = data['sentiment_category']
for _ in tqdm(range(1), desc="Model Training"):
    model, X_test, y_test = train_model(X, y)

# Step 5: Evaluate the model and save evaluation charts and reports
print("\nStep 5: Evaluating the model...")
for _ in tqdm(range(1), desc="Model Evaluation"):
    evaluate_model(model, X_test, y_test)


def categorize_sentiment(score):
    if score > 0:
        return 1  # Positive sentiment
    elif score < 0:
        return -1  # Negative sentiment
    else:
        return 0  # Neutral sentiment

def main():
    # User input for medication
    medication = input("Enter the drug name you are using (e.g., Mirtazapine): ").strip().lower()

    # Optional input for condition
    condition = input("Enter your condition (optional) (e.g., Depression): ").strip().lower()

    # Filter reviews based on medication and optionally the condition
    if condition:
        medication_reviews = data[(data['drugName'].str.lower() == medication) &
                                  (data['condition'].str.lower() == condition)]
    else:
        medication_reviews = data[data['drugName'].str.lower() == medication]

    if not medication_reviews.empty:
        sentiment = medication_reviews['sentiment'].mean()
        sentiment_category = categorize_sentiment(sentiment)
        print(f"Sentiment for {medication}: {sentiment_category}")
    else:
        print("No reviews found for the specified medication or condition.")

if __name__ == "__main__":
    main()


