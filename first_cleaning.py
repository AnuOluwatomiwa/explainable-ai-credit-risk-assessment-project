import pandas as pd
import numpy as np
import streamlit as st

def first_cleaning(df, credit_risk_scaler, one_hot_encoder):
    X=df.copy()
    credit_history = X.Credit_History
    X = X.drop("Credit_History", axis =1)
    
    # st.write(X)
    num_value = X.select_dtypes(include = np.number)
    cat_value = X.select_dtypes(exclude = np.number)

    def convert_to_0_1(row):
        return 1 if (row == 'Yes') or (row == 'Male') else 0

    cat_value['Married'] = cat_value['Married'].apply(convert_to_0_1)
    cat_value['Self_Employed'] = cat_value['Self_Employed'].apply(convert_to_0_1)
    cat_value['Gender'] = cat_value['Gender'].apply(convert_to_0_1)
    
    
    cat_values_ordinal = cat_value.loc[:, ['Gender', 'Married', 'Self_Employed']]
    cat_values_to_one_hot = cat_value.drop(['Gender', 'Married', 'Self_Employed'], axis = 1)
    
    # cat_values_to_one_hot = pd.get_dummies(cat_values_to_one_hot)
    cat_values_to_one_hot = one_hot_encoder.transform(cat_values_to_one_hot.astype("str"))
    
    cat_values_to_one_hot = pd.DataFrame(cat_values_to_one_hot.todense(), columns = one_hot_encoder.get_feature_names_out(), index=[1])
    
    perfect_cat_values = pd.concat([cat_values_ordinal, cat_values_to_one_hot], axis = 1)
    
    num_values_cols = num_value.columns

    num_values_arrays = credit_risk_scaler.transform(num_value)
    
    standardized_num_values = pd.DataFrame(num_values_arrays, columns = list(num_values_cols), index = num_value.index)
    
    restructured_X = pd.concat([standardized_num_values, perfect_cat_values, credit_history], axis = 1)
    
    return restructured_X