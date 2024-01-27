# import additional libraries
import pandas as pd
import numpy as np
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('Reddit_Data.csv')

# Handle missing values in 'clean_comment' column
df['clean_comment'].fillna('', inplace=True)

# Convert labels to numerical format
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['clean_comment'], df['category_encoded'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Split the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_vectorized, y_train, test_size=0.1, random_state=42)

# Build and compile the Neural Network model
model = Sequential()
model.add(Dense(128, input_dim=X_train_vectorized.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Neural Network model
model.fit(X_train_split, y_train_split, epochs=5, batch_size=64, validation_data=(X_val_split, y_val_split))

# Evaluate accuracy on the test set
_, accuracy = model.evaluate(X_test_vectorized, y_test)
print("Accuracy on Test Set:", accuracy)

# Print final classification report (you may need to adjust based on your problem)
y_pred_nn = model.predict_classes(X_test_vectorized)
print("\nFinal Classification Report (Neural Network):")
print(classification_report(y_test, y_pred_nn))
