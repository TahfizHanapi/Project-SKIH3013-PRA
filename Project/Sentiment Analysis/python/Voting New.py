from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from joblib import dump

# Load the dataset
df = pd.read_csv('C:\\Users\\USER\\Desktop\\PRA\\PRA Project\\static\\csv\\mazda.csv')


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


# Preprocess the 'headline' column
df['headline'] = df['headline'].apply(preprocess_text)

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


# Apply sentiment analysis to 'headline' column
df['sentiment'] = df['headline'].apply(get_sentiment)

# Convert 'label' and 'sentiment' columns to strings
df['label'] = df['label'].astype(str)
df['sentiment'] = df['sentiment'].astype(str)

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(df['label'], df['sentiment']))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['label'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Check if the test set is empty
if X_test_vectorized.shape[0] == 0:
    print("Error: Test set is empty. Check your train-test split or vectorization process.")
    exit()

# Support Vector Machine (SVM) Classifier
svm_params = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
svm_classifier = GridSearchCV(SVC(probability=True), svm_params, cv=5)
svm_classifier.fit(X_train_vectorized, y_train)
y_pred_svm = svm_classifier.predict(X_test_vectorized)

# Gradient Boosting Machines (GBM) Classifier
gbm_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5]}
gbm_classifier = GridSearchCV(GradientBoostingClassifier(), gbm_params, cv=5)
gbm_classifier.fit(X_train_vectorized, y_train)
y_pred_gbm = gbm_classifier.predict(X_test_vectorized)

# Neural Network (LSTM) Classifier
mlp_params = {'hidden_layer_sizes': [(100,), (50, 100, 50)], 'max_iter': [500, 1000]}
mlp_classifier = GridSearchCV(MLPClassifier(), mlp_params, cv=5)
mlp_classifier.fit(X_train_vectorized, y_train)
y_pred_mlp = mlp_classifier.predict(X_test_vectorized)

# Print the best parameters for SVM
print("Best Parameters for SVM:", svm_classifier.best_params_)

# Print the best parameters for GBM
print("Best Parameters for GBM:", gbm_classifier.best_params_)

# Print the best parameters for MLP
print("Best Parameters for MLP:", mlp_classifier.best_params_)

# Evaluate accuracy on the test set for each classifier
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_gbm = accuracy_score(y_test, y_pred_gbm)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

print("\nAccuracy on Test Set (SVM):", accuracy_svm)
print("\nAccuracy on Test Set (GBM):", accuracy_gbm)
print("\nAccuracy on Test Set (MLP):", accuracy_mlp)

# Save trained models
dump(vectorizer, 'tfidf_vectorizer.joblib')
dump(svm_classifier, 'svm_classifier.joblib')
dump(gbm_classifier, 'gbm_classifier.joblib')
dump(mlp_classifier, 'mlp_classifier.joblib')
