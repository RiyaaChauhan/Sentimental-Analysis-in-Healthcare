# import pandas as pd
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# # Download NLTK resources if not already downloaded
# # nltk.download('punkt')
# # nltk.download('stopwords')
# # Set of English stopwords
# stop_words = set(stopwords.words('english'))

# # Function to clean the text: remove punctuation, lowercase, tokenize, and remove stopwords
# def clean_text(text):
#     text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
#     text = text.lower()  # convert to lowercase
#     tokens = word_tokenize(text)  # tokenize text
#     filtered_tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
#     return ' '.join(filtered_tokens)

# # Load and preprocess data
# def load_and_preprocess_data(train_file, test_file):
#     # Load the dataset
#     trainDataset = pd.read_csv(train_file, sep='\t')
#     testDataset = pd.read_csv(test_file, sep='\t')

#     # Concatenate train and test dataset
#     data = pd.concat([trainDataset, testDataset])

#     # Rename columns for better understanding
#     data.columns = ['Id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount']

#     # Drop missing values
#     data.dropna(inplace=True)

#     # Apply text cleaning
#     data['clean_review'] = data['review'].apply(clean_text)

#     return data





import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
stop_words = set(stopwords.words('english'))

# Function to clean the text: remove punctuation, lowercase, tokenize, and remove stopwords
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = text.lower()  # convert to lowercase
    tokens = word_tokenize(text)  # tokenize text
    filtered_tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return ' '.join(filtered_tokens)

# Load and preprocess data
def load_and_preprocess_data(train_file, test_file):
    # Load the dataset
    trainDataset = pd.read_csv(train_file, sep='\t')
    testDataset = pd.read_csv(test_file, sep='\t')

    # Concatenate train and test dataset
    data = pd.concat([trainDataset, testDataset])

    # Rename columns for better understanding
    data.columns = ['Id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount']

    # Drop missing values
    data.dropna(inplace=True)

    # Apply text cleaning
    data['clean_review'] = data['review'].apply(clean_text)

    return data