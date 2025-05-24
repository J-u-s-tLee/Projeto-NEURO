from sklearn.preprocessing import StandardScaler
import pandas as pd

def Standardization(df, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df)
    else:
        scaled_array = scaler.transform(df)
    
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns)
    return scaled_df, scaler
