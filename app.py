import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():  # put application's code here
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    model = createModel()
    form_features = [x for x in request.form.values()]
    features = [np.array(form_features)]
    prediction = model.predict(features)
    if prediction == 1:
        result = 'You are at high risk of Stroke.  Please consult your physician.'
    else:
        result = 'Your risk of Stroke is low.'

    return render_template('index.html', prediction_text=format(result))


def createModel():
    # Create model
    mlModel = pickle.load(open('model/model.pkl', 'rb'))
    return mlModel


if __name__ == '__main__':
    app.run()
