import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv('stroke.csv')
df = pd.DataFrame(raw_data, columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke'])
df = df.astype({'gender': 'category', 'hypertension': 'category', 'heart_disease': 'category', 'ever_married': 'category', 'work_type': 'category', 'Residence_type': 'category', 'smoking_status': 'category'})
#OneHotEncoder(handle_unknown='ignore').fit_transform(df['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
X = df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]
y = df['stroke']


numeric_features = ['age', 'avg_glucose_level', 'bmi']
mumeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', mumeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(
    steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression())]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

clf.fit(X_train, y_train)

pickle.dump(clf, open('model.pkl', 'wb'))
