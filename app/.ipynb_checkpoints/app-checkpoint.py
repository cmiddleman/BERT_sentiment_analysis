from flask import Flask, render_template, url_for, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from src.bert_sentiment import BertAmazonSentiment

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html', title='haha')


@app.route('/amazon')
def amazon():
    return render_template('amazon.html')

@app.route('/results', methods=['POST'])
def results():
    review_text = request.form['text_input1']
    return render_template('results.html', review_text=review_text)

@app.route('/twitch')
def twitch():
    return render_template('twitch.html')

if __name__=="__main__":
    app.run(debug=True)