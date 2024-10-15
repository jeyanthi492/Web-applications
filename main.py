<<<<<<< HEAD
main.py
print("Hello world")
=======
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import pickle
#from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index_old.html')

@app.route('/predict', methods = ['POST'])
def predict():
    print("In function 1")
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    print("In function 2")
    output = model.predict(features_value)
    print("In function 3")
    if output[0] ==0:
        #print("output is 0, employee will not leave")
        return render_template('index1.html')
    elif output[0] ==1:
        #print("output is 1, employee will leave")
        return render_template('index2.html')

if __name__ =="__main__":
    app.run()
>>>>>>> master
