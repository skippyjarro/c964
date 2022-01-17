import pickle

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, OneHotEncoder

raw_data = pd.read_csv('model/stroke.csv')
df = pd.DataFrame(raw_data, columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
                                     'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke'])
df = df.astype(
    {'gender': 'category', 'hypertension': 'category', 'heart_disease': 'category', 'ever_married': 'category',
     'work_type': 'category', 'Residence_type': 'category', 'smoking_status': 'category'})
df = df.dropna()
X = df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
        'avg_glucose_level', 'bmi', 'smoking_status']]
y = df['stroke']
numeric_features = ['age', 'avg_glucose_level', 'bmi']
mumeric_transformer = Pipeline(steps=[('poly', PolynomialFeatures(degree=2)), ('scaler', StandardScaler())])

categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
                        'smoking_status']
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', mumeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ]
)

smt = SMOTE(random_state=42, sampling_strategy='minority')
lor = LogisticRegression(C=50, max_iter=1000)

clf = Pipeline(
    steps=[('preprocessor', preprocessor), ('smt', smt), ('lor', lor)]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50, stratify=y)

clf.fit(X_train, y_train)


def getAccuracyScore():
    return accuracy_score(y_test, clf.predict(X_test))


# print(clf.score(X_train, y_train))
# print(clf.score(X_test, y_test))
# confusion = confusion_matrix(y_test, clf.predict(X_test))
# print(confusion)

pickle.dump(clf, open('model.pkl', 'wb'))

'''features = pd.DataFrame([['Male', 67, 0, 1, 'Yes', 'Private', 'Urban',
                          228.69, 36.6, 'formerly smoked']],
                        columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
                                 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']) '''

# print(clf.predict(features))
