import pickle
import shap
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if 'model_info' not in st.session_state:
    redirect_button = st.button('Get Your Reading')
    if redirect_button:
        st.switch_page('App.py')

else:
        
    message = st.session_state["message"]
    st.success(f"{message}")

    @st.cache_data
    def get_data(x_train_filename, x_test_filename):
        x_train = pd.read_csv(x_train_filename, index_col="Loan_ID")
        x_test = pd.read_csv(x_test_filename, index_col="Loan_ID")
        return x_train, x_test
    
    @st.cache_data
    def sample_data(x, nsamples=100):
        return shap.sample(x, nsamples=nsamples)

    @st.cache_data
    def global_explainer(x_train, x_test, _model): 
        explainer = shap.KernelExplainer(_model.predict_proba, x_train)

        shap_values = explainer.shap_values(x_test)
        
        return shap_values
    
    model_info = st.session_state['model_info']
    model = st.session_state['model']
    
    x_train_filename = "data/x_train.csv"
    x_test_filename = "data/x_test.csv"
    
    x_train, x_test = get_data(x_train_filename, x_test_filename)

    sampled_x_train = sample_data(x_train, 100)
    sampled_x_test = sample_data(x_test, 50)

    shap_values = global_explainer(sampled_x_train, sampled_x_test, model)
    
    st.header("Importance of Each Features on the Models Decision")    

    shap.initjs()    
    
    print(f"Shap Values Shape: {shap_values.shape}")
    print(f"sampled_x_test Shape: {sampled_x_test.shape}")    
    
    shap_values_0 = shap_values[...,0] / np.max(np.abs(shap_values[...,0]))
    shap_values_1 = shap_values[...,1] / np.max(np.abs(shap_values[...,1]))
    
    
    try:
        fig, ax = plt.subplots(figsize=(8,8))
        shap.summary_plot(shap_values_0, sampled_x_test, plot_type='bar', show=False)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.set_xlim(0, 0.0005)
        st.pyplot(fig)
    except IndexError as e:
        print(f"IndexError: {e}")