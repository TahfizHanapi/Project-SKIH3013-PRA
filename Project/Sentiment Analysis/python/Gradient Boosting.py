# import libraries
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('Reddit_Data.csv')  # Replace 'your_dataset.csv' with the actual file path

# Handle NaN values in the 'clean_comment' column
df['clean_comment'].fillna('', inplace=True)

# Convert 'category' to numerical format using LabelEncoder
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['clean_comment'], df['category_encoded'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Build and train the XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train_vectorized, y_train)

# Predict on the test set
y_pred_xgb = xgb_model.predict(X_test_vectorized)

# Evaluate accuracy on the test set
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("Accuracy on Test Set (XGBoost):", accuracy_xgb)

# Print final classification report
print("\nFinal Classification Report (XGBoost):")
print(classification_report(y_test, y_pred_xgb))
