import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from scipy import stats

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Tokenize the text
    words = word_tokenize(text)

    # Remove stop words
    words = [word for word in words if word not in stopwords.words('english')]

    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)


def show_wordcloud(data, title=None):
    wordcloud = WordCloud(
        background_color='white',
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=42
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


# Load your data
df = pd.read_csv('C:\Users\Tahfiz\Documents\UUM\SEM 5\Pattern Recognition & Analysis\Project-SKIH3013-PRA\Project\Sentiment Analysis\python\Reddit_Data.csv')

# Fill NaN values with an empty string
df['clean_comment'] = df['clean_comment'].fillna('')

# Preprocess the text
# Show word cloud for each category
for category in df['category'].unique():
    show_wordcloud(df[df['category'] == category]['clean_comment'], title=f"Most frequent words in category {category}")

# Tokenize and pad sequences
tokenizer = Tokenizer()

tokenizer.fit_on_texts(df['clean_comment'])

X_seq = tokenizer.texts_to_sequences(df['clean_comment'])
X_pad = pad_sequences(X_seq)
y = df['category']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# SVM
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
print(classification_report(y_test, svm_predictions))

# XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
print(classification_report(y_test, xgb_predictions))

# LSTM
model_lstm = Sequential()
model_lstm.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=X_pad.shape[1]))
model_lstm.add(LSTM(100))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
lstm_predictions = model_lstm.predict_classes(X_test)
print(classification_report(y_test, lstm_predictions))

# Majority voting
predictions = np.array([svm_predictions, xgb_predictions, lstm_predictions])
majority_vote = stats.mode(predictions)[0]

print(f"Majority Vote Prediction: {majority_vote}")
