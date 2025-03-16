from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    use_fast=False
)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    tokenizer=tokenizer
)

app = Flask(__name__)

def analyze_sentiment(message):
    result = sentiment_pipeline(message)
    sentiment_label = result[0]['label'].lower()
    print("Sentiment result:", result)
    return sentiment_label

def get_response(message):
    sentiment = analyze_sentiment(message)
    if sentiment == "positive":
        response = "Que bom ouvir isso! Você parece estar em um ótimo humor!"
    elif sentiment == "negative":
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
