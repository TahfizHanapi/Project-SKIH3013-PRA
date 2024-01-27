#!/usr/bin/env python
# coding: utf-8

# import libraries
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
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

# Define a smaller parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1],  # Regularization parameter
    'kernel': ['linear'],  # Kernel type (linear for efficiency)
}

# Initialize the SVM classifier
svm_classifier = SVC()

# Use GridSearchCV to perform hyperparameter tuning with parallelization
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Check for errors during hyperparameter tuning
try:
    grid_search.fit(X_train_vectorized, y_train)
except Exception as e:
    print("Error during hyperparameter tuning:", e)
    exit()

# Print the best hyperparameters
best_params = grid_search.best_params_
print("\nBest Hyperparameters:", best_params)

# Use the best hyperparameters to train the final model
best_svm_classifier = grid_search.best_estimator_
y_pred = best_svm_classifier.predict(X_test_vectorized)

# Evaluate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set:", accuracy)

# Print final classification report
print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred))

# User input loop
while True:
    # Get user input
    new_comment = input("Enter a new comment (type 'exit' to stop): ")

    # Check if the user wants to exit the loop
    if new_comment.lower() == 'exit':
        break

    # Preprocess the new comment
    preprocessed_comment = preprocess_text(new_comment)

    # Get sentiment prediction for the preprocessed comment
    new_sentiment = get_sentiment(preprocessed_comment)

    # Print or use the predicted sentiment
    print("Predicted Sentiment:", new_sentiment)
    print()
