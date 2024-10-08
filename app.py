from flask import Flask, request, jsonify, render_template
import pickle
import spacy
import re

# Flask app create karte hain
app = Flask(__name__)

# Load the saved model
with open('sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the saved vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Load Spacy model
nlp = spacy.load('en_core_web_sm')

# Preprocess function
def preprocess_review(review):
    review = review.lower()
    review = re.sub(r'[^\w\s]', '', review)
    tokens = [token.text for token in nlp(review) if not nlp.vocab[token.text].is_stop]
    return " ".join(tokens)

# Home page
@app. route('/')
def home():
    return render_template('index.html')

# API route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # JSON data lo
    review = data['review']  # review extract kar
    cleaned_review = preprocess_review(review)
    review_vec = vectorizer.transform([cleaned_review])
    prediction = model.predict(review_vec)
    sentiment = 'positive' if prediction[0] == 1 else 'negative'
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)