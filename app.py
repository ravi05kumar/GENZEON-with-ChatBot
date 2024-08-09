from flask import Flask, render_template, request, jsonify
import random
import string
import numpy as np
import warnings
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Load and preprocess corpus
with open('corpus.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

# Tokenize corpus into sentences
sent_tokens = nltk.sent_tokenize(corpus)

# Prepare punctuation removal dictionary
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    """
    Normalize text by lowercasing and removing punctuation.
    """
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))

# Greeting inputs and responses
greeting_input = ["hi", "hello", "hey", "hola", "namaste"]
greeting_response = ["howdy", "hey there", "hi", "hello :)"]

def greeting(sentence):
    """
    Return a greeting response if the input sentence contains a greeting word.
    """
    for word in sentence.split():
        if word.lower() in greeting_input:
            return random.choice(greeting_response)
    return None

def response(user_response):
    """
    Generate a response based on the user's input using TF-IDF and cosine similarity.
    """
    user_response = user_response.lower()
    robo_response = ''
    sent_tokens.append(user_response)
    tfidfvec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = tfidfvec.fit_transform(sent_tokens)
    val = cosine_similarity(tfidf[-1], tfidf[:-1])  # Compare with all except the last added
    idx = val.argsort()[0][-1]
    flat = val.flatten()
    flat.sort()
    score = flat[-1]
    if score == 0:
        robo_response = "Sorry, I don't understand"
    else:
        robo_response = sent_tokens[idx]
    sent_tokens.pop()  # Remove the last added user_response
    return robo_response

@app.route("/")
def home():
    """
    Render the home page.
    """
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    """
    Handle user input and return the chatbot's response.
    """
    user_input = request.form["user_input"]
    if greeting(user_input) is not None:
        return jsonify({"response": greeting(user_input)})
    else:
        return jsonify({"response": response(user_input)})

if __name__ == "__main__":
    app.run(debug=True)
