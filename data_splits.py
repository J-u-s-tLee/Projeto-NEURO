import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath='unsupervised_output/labelled_data.csv', label_column='class', train_size=0.7, val_size=0.2, test_size=0.1, random_state=42):
        
        df = pd.read_csv(filepath)
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        X = df.drop(columns=[label_column])
        y = df[label_column]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1.0 - train_size), random_state=random_state)

        val_relative_size = val_size / (val_size + test_size)

        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1.0 - val_relative_size), random_state=random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test
