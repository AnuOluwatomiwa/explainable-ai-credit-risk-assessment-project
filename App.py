import pickle

import streamlit as st

import numpy as np
import pandas as pd

from first_cleaning import first_cleaning
from engineering import engineering

st.set_page_config(page_title='LAAD', layout = 'wide', page_icon='')
left_col, center_col, right_col = st.columns((1,2,1))

with center_col:
    st.header('Loan Application Approval Decider')

df_filename = "data/copy_loan_cleaned.xlsx"
model_filename="models_and_encoders/best_loan_model(SVC).pkl"
crs_filename="models_and_encoders/credit_risk_scaler.pkl"
ncs_filename="models_and_encoders/new_col_scaler.pkl"

@st.cache_data
def get_used_df(df_filename):
    return pd.read_excel(df_filename)

@st.cache_resource
def get_model(model_filename:str, 
            cr_scaler_filename:str,
            nc_scaler_filename:str):
    credit_risk_scaler = pickle.load(open(cr_scaler_filename, 'rb'))
    new_col_scaler = pickle.load(open(nc_scaler_filename, 'rb'))
    model = pickle.load(open(model_filename, 'rb'))
    return model, credit_risk_scaler, new_col_scaler

model, credit_risk_scaler, new_col_scaler = get_model(model_filename, crs_filename, ncs_filename)
df = get_used_df(df_filename)

with center_col:
    gender = st.selectbox("Gender", options=df.Gender.unique())
    married = st.selectbox("Marital Status", options=df.Married.unique())
    dependents = st.selectbox("Number Of Dependents", options=df.Dependents.unique())
    education = st.selectbox("Education Status", options=df.Education.unique())
    self_employed = st.selectbox("Are You Self Employed", options=df.Self_Employed.unique())
    applicant_income = st.number_input("Applicant Income", min_value=0.0)
    coapplicant_income = st.number_input("Co-Applicant Inocome", min_value=0.0)
    loan_amount = st.number_input("Loan Amount", min_value=1.0)
    loan_amt_term = st.number_input("Loan Duration (IN MONTHS)", min_value=np.array(df.Loan_Amount_Term).min(), max_value=np.array(df.Loan_Amount_Term).max())
    credit_history = st.selectbox("Credit History", options=['Yes', 'No'])
    property_area = st.selectbox('Where Do You Reside', options=df.Property_Area.unique())
    button = st.button("Loan Eligible?")
    
    if button:
        infos = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_amt_term,
            "Credit_History": credit_history,
            "Property_Area": property_area,
        }
        
        infos_df = pd.DataFrame(infos, index = [1], columns=df.columns[1:-1])
        
        # cleaned_df = first_cleaning(infos_df, credit_risk_scaler)
        # engineered_df = engineering(cleaned_df, new_col_scaler)
        
        # st.table(engineered_df)

        st.success("We're Almost There!!! ")