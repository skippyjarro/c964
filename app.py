import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

raw_data = pd.read_csv('model/stroke.csv')
df = pd.DataFrame(raw_data, columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
                                     'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke'])
df = df.astype(
    {'gender': 'category', 'hypertension': 'category', 'heart_disease': 'category', 'ever_married': 'category',
     'work_type': 'category', 'Residence_type': 'category', 'smoking_status': 'category'})
X = df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
        'avg_glucose_level', 'bmi', 'smoking_status']]
y = df['stroke']

# create model
model = pickle.load(open('model/model.pkl', 'rb'))


@app.route('/', methods=['GET'])
def index():  # put application's code here
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form.get('Sex')
    age = int(request.form.get('Age'))
    hypertension = int(request.form.get('Hypertension'))
    heart_disease = int(request.form.get('Heart_Disease'))
    ever_married = request.form.get('Ever_Married')
    work_type = request.form.get('Work_Type')
    residence_type = request.form.get('Residence_type')
    avg_glucose_level = request.form.get('avg_glucose_level')
    if avg_glucose_level == '':
        avg_glucose_level = 85
    else:
        avg_glucose_level = float(avg_glucose_level)
    bmi = int(request.form.get('bmi'))
    smoking_status = request.form.get('Smoking_Status')
    features = pd.DataFrame([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type,
                              avg_glucose_level, bmi, smoking_status]],
                            columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
                                     'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])
    prediction = model.predict(features)
    if prediction == 1:
        result = 'You are at high risk of Stroke.  Please consult your physician.'
    else:
        result = 'Your risk of Stroke is low.'

    return render_template('index.html', prediction_text=format(result))


@app.route('/dashboard', methods=['POST', 'GET'])
def dashboard():
    return render_template('dashboard.html')


@app.route('/graph', methods=['POST'])
def graph():
    feature = [request.form.get('dashboard_type')]
    if feature is None:
        feature = ['gender']
    labels = [label for label in np.unique(df[feature])]
    dataset = df[feature].value_counts()
    data = []
    for label in labels:
        data.append(dataset.get(label))
    return render_template('dashboard.html', labels=labels, data=data, feature=feature)


if __name__ == '__main__':
    app.run()
