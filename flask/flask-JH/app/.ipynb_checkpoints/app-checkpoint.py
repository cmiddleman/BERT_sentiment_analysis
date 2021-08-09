from flask import Flask, render_template, url_for, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html', title='haha')


@app.route('/amazon')
def amazon():
    return render_template('amazon.html')

@app.route('/twitch')
def twitch():
    return render_template('twitch.html')

if __name__=="__main__":
    app.run(debug=True)