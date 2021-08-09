from flask import Flask, render_template, url_for, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from src.bert_sentiment import BertAmazonSentiment

app = Flask(__name__)

pkl_path = 'models/amazon_reviews/'
bert_amazon = BertAmazonSentiment(pkl_path)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/amazon')
def amazon():
    return render_template('amazon.html')

@app.route('/results', methods=['POST'])
def results():
    review_text = request.form['text_input1']
    sentiment = bert_amazon.pretty_classify_one(review_text)
    return render_template('results.html', bert_sentiment=sentiment)

@app.route('/twitch')
def twitch():
    return render_template('twitch.html')

if __name__=="__main__":
    app.run(debug=True)