import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


st.set_page_config(page_title="Customer-Churn-and-Salary-Prediction", layout="wide")
st.title("Customer-Churn-and-Salary-Prediction")


tab1, tab2 = st.tabs(
    [
        "Customer Churn Classification",
        "Salary Prediction Regression"
    ]
)

#Customer Churn Classification

with tab1:
    st.header("Customer Churn Classification")
    st.write("This tab allows you to predict whether a customer will churn based on various input features.")

    # Load the trained model
    model = tf.keras.models.load_model('model.h5')

    # Load the encoders and scaler
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)


    ## streamlit app

    # User input
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0], key='geography_churn')
    gender = st.selectbox('Gender', label_encoder_gender.classes_, key='gender_churn')
    age = st.slider('Age', 18, 92, key='age_churn')
    balance = st.number_input('Balance', key='balance_churn')
    credit_score = st.number_input('Credit Score', key='credit_score_churn')
    estimated_salary = st.number_input('Estimated Salary', key='estimated_salary_churn')
    tenure = st.slider('Tenure', 0, 10, key='tenure_churn')
    num_of_products = st.slider('Number of Products', 1, 4, key='num_of_products_churn')
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], key='has_cr_card_churn')
    is_active_member = st.selectbox('Is Active Member', [0, 1], key='is_active_member_churn')

    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)


    # Predict churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.write(f'Churn Probability: {prediction_proba:.2f}')

    if prediction_proba > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')



#Salary Prediction Regression

with tab2:
    st.header("Salary Prediction Regression")
    st.write("This tab allows you to predict the estimated salary of a customer based on various input features.")

    # Load the trained model
    model = tf.keras.models.load_model('regression_model.h5')

    # Load the encoders and scaler
    with open('label_encoder_gender_regression.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('onehot_encoder_geo_regression.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('scaler_regression.pkl', 'rb') as file:
        scaler = pickle.load(file)


    # User input
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0], key='geography_regression')
    gender = st.selectbox('Gender', label_encoder_gender.classes_, key='gender_regression')
    age = st.slider('Age', 18, 92, key='age_regression')
    balance = st.number_input('Balance', key='balance_regression')
    credit_score = st.number_input('Credit Score', key='credit_score_regression')
    exited = st.selectbox('Exited', [0, 1], key='exited_regression')
    tenure = st.slider('Tenure', 0, 10, key='tenure_regression')
    num_of_products = st.slider('Number of Products', 1, 4, key='num_of_products_regression')
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], key='has_cr_card_regression')
    is_active_member = st.selectbox('Is Active Member', [0, 1], key='is_active_member_regression')


    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'Exited': [exited]
    })


    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)


    # Predict estimated salary
    prediction = model.predict(input_data_scaled)
    prediction_salary = prediction[0][0]

    st.write(f'Predicted Estimated Salary: ${prediction_salary:.2f}')