from flask import Flask, request 
import pandas as pd
import numpy as np 
import pickle 
import flasgger
from flasgger import Swagger


app = Flask(__name__)
Swagger(app) # we need to initialize the swagger app.

pickle_in = open('classifier.pkl', 'rb')
classifier =  pickle.load(pickle_in)

@app.route('/') 
def welcome():
    return "Welcome all" 

@app.route('/predict') 
def predict_note_aunthentication():
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    predictions = classifier.predict( [[variance, skewness, curtosis, entropy]] )
    return f"The Predicted value is: {predictions}"

# You should give the values like this http://127.0.0.1:5000/predict?variance=2&skewness=3&curtosis=2&entropy=1


if __name__ == '__main__':
    app.run()