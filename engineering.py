import pandas as pd
import numpy as np

def engineering(df, new_col_scaler):
    cleaned_df = df.copy()
    total_income = cleaned_df['ApplicantIncome'] + cleaned_df['CoapplicantIncome']
    loan_amt_to_total_income_ratio = cleaned_df['LoanAmount'] / total_income
    
    dependents = cleaned_df['Dependents'].replace('3+', 3).astype(float)

    total_family_members = dependents + 1

    # Calculate Dependents Ratio
    dependents_ratio = dependents / total_family_members
    
    new_columns = {
        'TotalIncome': total_income,
        'LoanAmtToTotalIncomeRatio': loan_amt_to_total_income_ratio,
        'DependentsRatio': dependents_ratio 
    }

    new_columns_df = pd.DataFrame(new_columns)
    new_columns_scaled = new_col_scaler.transform(new_columns_df)
    new_columns_scaled_df = pd.DataFrame(data = new_columns_scaled, index = new_columns_df.index, columns = new_columns_df.columns)
    
    full_df_engineered = pd.concat([new_columns_scaled_df, cleaned_df], axis = 1)
    
    return full_df_engineered