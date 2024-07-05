#Import the LimeTabularExplainer module
import numpy as np
import streamlit as st
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from rescale import rescale

df_filename = "/workspaces/explainable-ai-credit-risk-assessment-project/data/dataset_v3.xlsx"

if 'model_info' not in st.session_state:
    redirect_button = st.button('Get Your Reading')
    if redirect_button:
        st.switch_page('App.py')
        
else:
    
    message = st.session_state["message"]
    st.success(f"{message}")

    @st.cache_data
    def get_used_df(df_filename):
        return pd.read_excel(df_filename)
    
    
    new_col_scaler = st.session_state['new_col_scaler']
    credit_risk_scaler = st.session_state['credit_risk_scaler']
    
    engineered_info = get_used_df(df_filename)
    engineered_info.drop('Loan_Status', axis=1, inplace=True)
    
    eng_df_rescaled = rescale(engineered_info, new_col_scaler, credit_risk_scaler)
    
    model = st.session_state['model']
    instance = st.session_state['model_info']
    prediction = st.session_state['pred']

    instance_scaled = rescale(instance, new_col_scaler, credit_risk_scaler)

    # Get the feature names
    feature_names = list(engineered_info.columns)

    class_names = [0, 1]

    explainer = LimeTabularExplainer(eng_df_rescaled.values,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    mode='classification', random_state=42)


    if isinstance(prediction, (list, np.ndarray)):
        prediction = prediction[0] 

    exp = explainer.explain_instance(np.array(instance_scaled).flatten(), model.predict_proba, num_features=20, labels=(prediction,))
# Optionally, display the LIME explanation plot
    st.write("LIME Explanation Plot:")
    fig = exp.as_pyplot_figure(label=prediction)
    st.pyplot(fig)