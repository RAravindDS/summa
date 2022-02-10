from flask import Flask, request 
import pandas as pd
import numpy as np 
import pickle 

# when you want to start the flask app, you need to start from specific point~! 

app = Flask(__name__) #""" This is where our flask app starts """ # This is the first step of the flask app. 
pickle_in = open('classifier.pkl', 'rb')
classifier =  pickle.load(pickle_in)

@app.route('/') # root path decorator.
def welcome():
    return "Welcome all" 

@app.route('/predict') # It is usually a Get Method. 
def predict_note_aunthentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    predictions = classifier.predict( [[variance, skewness, curtosis, entropy]] )
    return f"The Predicted value is: {predictions}"

# You should give the values like this http://127.0.0.1:5000/predict?variance=2&skewness=3&curtosis=2&entropy=1


if __name__ == '__main__':
    app.run()