from sklearn.preprocessing import StandardScaler
import pandas as pd

def Standardization(df):
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns)

    return scaled_df
