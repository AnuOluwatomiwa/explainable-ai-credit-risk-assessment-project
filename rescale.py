def rescale(df, new_col_scaler, credit_risk_scaler):
    df_copied = df.copy()
    df_copied.iloc[:, :3] = new_col_scaler.inverse_transform(df_copied.iloc[:, :3])
    df_copied.iloc[:, 3:7] = credit_risk_scaler.inverse_transform(df_copied.iloc[:, 3:7])
    
    return df_copied