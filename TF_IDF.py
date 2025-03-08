from sklearn.feature_extraction.text import TfidfVectorizer

# Function to apply TF-IDF vectorizer on the cleaned reviews
def apply_tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(data['clean_review'])
    return X, tfidf_vectorizer
