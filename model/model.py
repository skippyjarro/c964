from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logR = LogisticRegression()
le = LabelEncoder()
raw_data = pd.read_csv('stroke.csv')
df = pd.DataFrame(raw_data, columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke'])
cleaned_df = df.apply(le.fit_transform)
X = cleaned_df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]
y = cleaned_df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
pickle.dump(classifier, open('model.pkl', 'wb'))