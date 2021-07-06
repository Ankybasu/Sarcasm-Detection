from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
loaded_model = pickle.load(open('smodel.pkl', 'rb'))
tfvect = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=150000)
dataframe = pd.read_csv("train-balanced-sarcasm.csv")
x = tfvect.fit_transform(dataframe['comment'].apply(lambda x: np.str_(x)))
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def requestResults(result):
    if result == 0:
        return "Not-Sarcastic"
    else:
        return "Sarcastic"
def sardet(text): 
    input_data = [text]
    vectorized_input_data =  tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    message = request.form['message']
    pred = sardet(message)
    result=requestResults(pred)
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
