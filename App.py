import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import pickle
#from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle


model=tf.keras.models.load_model(r'C:\Users\sunil\LEARNING\Churn_prediction\churn_model.h5')

with open("le_gen.pkl","rb") as f1:
    le_gen=pickle.load(f1)
with open("ohe_geo.pkl","rb") as f2:
    ohe_geo=pickle.load(f2)

with open("scaler.pkl","rb") as f3:
    scaler=pickle.load(f3)


st.title("CUSTOMER CHURN PREDICTION MODEL")

# User input
geography = st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender', le_gen.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [le_gen.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = ohe_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))



# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]



if st.button("PREDICT"):
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]
    st.write(f'Churn Probability: {prediction_proba:.2f}')
    if prediction_proba > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')