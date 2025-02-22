import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_data():
    data = pd.read_csv(r"C:\Users\om\Downloads\loan_approval_dataset.csv")
    data['loan_status'] = data['loan_status'].map({'Approved': 1, 'Rejected': 0})
    return data

data = load_data()

st.write("Loan Approval Data Sample")
st.write(data.head())

label_enc = LabelEncoder()
data['education'] = label_enc.fit_transform(data['education'])
data['self_employed'] = label_enc.fit_transform(data['self_employed'])

X = data.drop(columns=['loan_id', 'loan_status'])
y = data['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.title("Loan Approval Prediction")

st.write("## Model Accuracy: ", accuracy)

st.write("### Enter Applicant Details:")
input_data = {}
for col in X.columns:
    if data[col].dtype == 'object':
        options = data[col].unique()
        input_data[col] = st.selectbox(col, options)
    else:
        input_data[col] = st.number_input(col, min_value=float(X[col].min()), max_value=float(X[col].max()))


input_df = pd.DataFrame([input_data])


if st.button("Predict Loan Approval"):
    prediction = model.predict(input_df)
    result = "Approved" if prediction[0] == 1 else "Rejected"
    st.write(f"### Loan Prediction: {result}")


