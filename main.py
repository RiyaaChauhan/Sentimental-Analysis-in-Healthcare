# import pandas as pd
# from preprocessing import load_and_preprocess_data
# from sentiment_labeling import apply_sentiment_labeling
# from TF_IDF import apply_tfidf
# from Model import train_model
# from evaluation import evaluate_model
# from tqdm import tqdm
# import nltk

# # # Download NLTK resources if not already downloaded
# # nltk.download('punkt')
# # nltk.download('stopwords')
# # nltk.download('vader_lexicon')

# # File paths for the train and test dataset
# train_file = "drugsComTrain_raw.tsv"
# test_file = "drugsComTest_raw.tsv"

# # Step 1: Load and preprocess the data
# print("Step 1: Loading and preprocessing the data...")
# for _ in tqdm(range(1), desc="Loading & Preprocessing"):
#     data = load_and_preprocess_data(train_file, test_file)

# # Step 2: Apply sentiment labeling
# print("\nStep 2: Applying sentiment labeling...")
# for _ in tqdm(range(1), desc="Sentiment Labeling"):
#     data = apply_sentiment_labeling(data)

# # Step 3: Apply TF-IDF vectorization
# print("\nStep 3: Applying TF-IDF vectorization...")
# for _ in tqdm(range(1), desc="TF-IDF Vectorization"):
#     X, tfidf_vectorizer = apply_tfidf(data)

# # Step 4: Train the sentiment classification model
# print("\nStep 4: Training the sentiment classification model...")
# y = data['sentiment_category']
# for _ in tqdm(range(1), desc="Model Training"):
#     model, X_test, y_test = train_model(X, y)

# # Step 5: Evaluate the model and save evaluation charts and reports
# print("\nStep 5: Evaluating the model...")
# for _ in tqdm(range(1), desc="Model Evaluation"):
#     evaluate_model(model, X_test, y_test)


# def categorize_sentiment(score):
#     if score > 0:
#         return 1  # Positive sentiment
#     elif score < 0:
#         return -1  # Negative sentiment
#     else:
#         return 0  # Neutral sentiment

# def main():
#     # User input for medication
#     medication = input("Enter the drug name you are using (e.g., Mirtazapine): ").strip().lower()

#     # Optional input for condition
#     condition = input("Enter your condition (optional) (e.g., Depression): ").strip().lower()

#     # Filter reviews based on medication and optionally the condition
#     if condition:
#         medication_reviews = data[(data['drugName'].str.lower() == medication) &
#                                   (data['condition'].str.lower() == condition)]
#     else:
#         medication_reviews = data[data['drugName'].str.lower() == medication]

#     if not medication_reviews.empty:
#         sentiment = medication_reviews['sentiment'].mean()
#         sentiment_category = categorize_sentiment(sentiment)
#         print(f"Sentiment for {medication}: {sentiment_category}")
#     else:
#         print("No reviews found for the specified medication or condition.")

# if __name__ == "__main__":
#     main()



import pandas as pd
from preprocessing import load_and_preprocess_data
from sentiment_labeling import apply_sentiment_labeling
from TF_IDF import apply_tfidf
from Model import train_model
from evaluation import evaluate_model
from tqdm import tqdm
import nltk
import html
import re

# Ensure NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

# File paths for the train and test dataset
train_file = "drugsComTrain_raw.tsv"
test_file = "drugsComTest_raw.tsv"

# Function to clean the text
def clean_text(text):
    if isinstance(text, str):  # Ensure the input is a string
        # Decode HTML entities (like &#039;)
        text = html.unescape(text)
        # Remove non-alphanumeric characters and extra spaces
        text = re.sub(r'[^\w\s]', '', text)
        text = text.strip()  # Remove leading/trailing spaces
        return text
    else:
        return ''  # If it's not a string, return an empty string

# Step 1: Load and preprocess the data
print("Step 1: Loading and preprocessing the data...")
def load_and_preprocess_data(train_file, test_file):
    # Load data
    data = pd.read_csv(train_file, sep="\t", encoding="ISO-8859-1")

    # Check and handle missing reviews
    data['review'] = data['review'].fillna('')  # Replace NaN reviews with empty string
    data['clean_review'] = data['review'].apply(clean_text)

    return data

for _ in tqdm(range(1), desc="Loading & Preprocessing"):
    data = load_and_preprocess_data(train_file, test_file)

# Step 2: Apply sentiment labeling
print("\nStep 2: Applying sentiment labeling...")
def apply_sentiment_labeling(data):
    # Add sentiment labeling logic here (positive/negative/neutral)
    data['sentiment_category'] = data['rating'].apply(lambda x: 1 if x > 3 else -1 if x < 3 else 0)
    return data

for _ in tqdm(range(1), desc="Sentiment Labeling"):
    data = apply_sentiment_labeling(data)

# Step 3: Apply TF-IDF vectorization
print("\nStep 3: Applying TF-IDF vectorization...")
from sklearn.feature_extraction.text import TfidfVectorizer

def apply_tfidf(data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X = tfidf_vectorizer.fit_transform(data['clean_review'])
    return X, tfidf_vectorizer

for _ in tqdm(range(1), desc="TF-IDF Vectorization"):
    X, tfidf_vectorizer = apply_tfidf(data)

# Step 4: Train the sentiment classification model
print("\nStep 4: Training the sentiment classification model...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, X_test, y_test

y = data['sentiment_category']
for _ in tqdm(range(1), desc="Model Training"):
    model, X_test, y_test = train_model(X, y)

# Step 5: Evaluate the model and save evaluation charts and reports
print("\nStep 5: Evaluating the model...")
from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

for _ in tqdm(range(1), desc="Model Evaluation"):
    evaluate_model(model, X_test, y_test)

# Sentiment categorization function
def categorize_sentiment(score):
    if score > 0:
        return 1  # Positive sentiment
    elif score < 0:
        return -1  # Negative sentiment
    else:
        return 0  # Neutral sentiment

# Main function for user interaction
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
        sentiment = medication_reviews['sentiment_category'].mean()
        sentiment_category = categorize_sentiment(sentiment)
        print(f"Sentiment for {medication}: {sentiment_category}")
    else:
        print("No reviews found for the specified medication or condition.")

if __name__ == "__main__":
    main()
