import pickle

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():  # put application's code here
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    model = createModel()
    gender = request.form.get('Sex')
    age = int(request.form.get('Age'))
    hypertension = int(request.form.get('Hypertension'))
    heart_disease = int(request.form.get('Heart_Disease'))
    ever_married = request.form.get('Ever_Married')
    work_type = request.form.get('Work_Type')
    residence_type = request.form.get('Residence_type')
    avg_glucose_level = float(request.form.get('avg_glucose_level'))
    bmi = int(request.form.get('bmi'))
    smoking_status = request.form.get('Smoking_Status')
    features = pd.DataFrame([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]], columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])
    prediction = model.predict(features)
    print(prediction)
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
