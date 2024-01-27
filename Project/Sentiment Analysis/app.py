from flask import Flask, render_template, request, jsonify
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load the existing data
df = pd.read_csv('python/Reddit_Data.csv')

# Initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def preprocess_text(text):
    if isinstance(text, str):
        tokens = word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        processed_text = ' '.join(lemmatized_tokens)
        return processed_text

def get_sentiment(text):
    if isinstance(text, str):
        scores = analyzer.polarity_scores(text)
    else:
        scores = analyzer.polarity_scores(str(text))

    compound_score = scores['compound']

    if compound_score >= 0.05:
        sentiment = "Positive"  # positive
    elif compound_score <= -0.05:
        sentiment = "Negative"  # negative
    else:
        sentiment = "Neutral"  # neutral

    return sentiment

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    user_input = request.form['user_input']
    preprocessed_comment = preprocess_text(user_input)
    new_sentiment = get_sentiment(preprocessed_comment)
    return jsonify({'predicted_sentiment': new_sentiment})

# New route for the result page
@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
