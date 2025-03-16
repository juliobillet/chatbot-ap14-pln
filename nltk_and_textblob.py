from flask import Flask, request, jsonify, render_template
from textblob import TextBlob
import nltk

nltk.download('punkt')

app = Flask(__name__)

def analyze_sentiment(message):
    blob = TextBlob(message)
    return blob.sentiment.polarity

def get_response(message):
    polarity = analyze_sentiment(message)
    if polarity > 0.05:
        response = "Que bom ouvir isso! Você parece estar em um ótimo humor!"
    elif polarity < -0.05:
        response = "Sinto muito que você esteja se sentindo mal. Quer conversar sobre isso?"
    else:
        response = "Entendo. Pode me contar mais sobre o que está acontecendo?"
    return response

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/get_response', methods=['POST'])
def chat_response():
    user_message = request.form['message']
    response = get_response(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
