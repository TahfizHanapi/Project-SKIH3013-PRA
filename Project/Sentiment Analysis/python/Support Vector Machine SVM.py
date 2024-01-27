#!/usr/bin/env python
# coding: utf-8

# import libraries
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load the dataset
df = pd.read_csv('Reddit_Data.csv')

# Function for text preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        tokens = word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    else:
        tokens = word_tokenize(str(text))
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# Preprocess the 'clean_comment' column
df['clean_comment'] = df['clean_comment'].apply(preprocess_text)

# Initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment using NLTK
def get_sentiment(text):
    if isinstance(text, str):
        scores = analyzer.polarity_scores(text)
    else:
        scores = analyzer.polarity_scores(str(text))

    compound_score = scores['compound']

    if compound_score >= 0.05:
        sentiment = 1  # positive
    elif compound_score <= -0.05:
        sentiment = -1  # negative
    else:
        sentiment = 0  # neutral

    return sentiment

# Apply sentiment analysis to 'clean_comment' column
df['sentiment'] = df['clean_comment'].apply(get_sentiment)

# Convert 'category' and 'sentiment' columns to strings
df['category'] = df['category'].astype(str)
df['sentiment'] = df['sentiment'].astype(str)

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(df['category'], df['sentiment']))

# Print classification report
print("Classification Report:")
print(classification_report(df['category'], df['sentiment']))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['clean_comment'], df['category'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Check if the test set is empty
if X_test_vectorized.shape[0] == 0:
    print("Error: Test set is empty. Check your train-test split or vectorization process.")
    exit()

# Support Vector Machine (SVM) Classifier
svm_classifier = SVC()
svm_classifier.fit(X_train_vectorized, y_train)
y_pred_svm = svm_classifier.predict(X_test_vectorized)

# Evaluate accuracy on the test set
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("\nAccuracy on Test Set (SVM):", accuracy_svm)

# Print final classification report
print("\nFinal Classification Report (SVM):")
print(classification_report(y_test, y_pred_svm))

# Naive Bayes Classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)
y_pred_nb = nb_classifier.predict(X_test_vectorized)

# Evaluate accuracy on the test set
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("\nAccuracy on Test Set (Naive Bayes):", accuracy_nb)

# Print final classification report
print("\nFinal Classification Report (Naive Bayes):")
print(classification_report(y_test, y_pred_nb))

# User input loop
while True:
    # Get user input
    new_comment = input("Enter a new comment (type 'exit' to stop): ")

    # Check if the user wants to exit the loop
    if new_comment.lower() == 'exit':
        break

    # Preprocess the new comment
    preprocessed_comment = preprocess_text(new_comment)

    # Vectorize the new comment using the trained TF-IDF vectorizer
    new_comment_vectorized = vectorizer.transform([preprocessed_comment])

    # Predict using SVM
    new_comment_pred_svm = svm_classifier.predict(new_comment_vectorized)
    print("Predicted Sentiment (SVM):", new_comment_pred_svm[0])

    # Predict using Naive Bayes
    new_comment_pred_nb = nb_classifier.predict(new_comment_vectorized)
    print("Predicted Sentiment (Naive Bayes):", new_comment_pred_nb[0])

    print()
