from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)

loaded_model = pickle.load(open('smodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    return render_template('index.html', prediction_text="Sarcastic")

if __name__ == '__main__':
    app.run(debug=True,threaded=True)
