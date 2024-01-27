#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# In[2]:


# Load the dataset
df = pd.read_csv('Reddit_Data.csv')


# In[3]:


# Preprocessing
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(str(text))

    # Lowercasing
    tokens = [word.lower() for word in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


# Assuming that the text column is the first column in your DataFrame
df['processed_text'] = df.iloc[:, 0].apply(preprocess_text)

# In[4]:


# Feature Extraction
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(df['processed_text'])

# In[5]:


# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
    'fit_prior': [True, False]
}

# In[6]:


# Model Selection
X_train, X_test, y_train, y_test = train_test_split(features, df.iloc[:, 1], test_size=0.2, random_state=42)
model = MultinomialNB()

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# In[7]:


# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters: ", best_params)

# Get the best score
best_score = grid_search.best_score_
print("Best Score: ", best_score)

# Predict on the test set
predictions = grid_search.predict(X_test)

# In[8]:


# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: ", accuracy)
precision = precision_score(y_test, predictions, average='weighted')
print("Precision: ", precision)
recall = recall_score(y_test, predictions, average='weighted')
print("Recall: ", recall)
f1 = f1_score(y_test, predictions, average='weighted')
print("F1 Score: ", f1)

# In[9]:


# initialize NLTK sentiment analyzer

analyzer = SentimentIntensityAnalyzer()


# In[10]:


def get_sentiment(text):
    # Check if the input is a string
    if isinstance(text, str):
        scores = analyzer.polarity_scores(text)
    else:
        # Handle non-string input here
        # For example, you might convert floats to strings:
        scores = analyzer.polarity_scores(str(text))

    # Get compound score
    compound_score = scores['compound']

    # Determine sentiment based on compound score
    if compound_score >= 0.05:
        sentiment = 1  # positive
    elif compound_score <= -0.05:
        sentiment = -1  # negative
    else:
        sentiment = 0  # neutral

    return sentiment


# In[ ]:


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
