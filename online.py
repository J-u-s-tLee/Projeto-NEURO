import buffer
import threading
import joblib
import signal_processing
import feature_extraction
import pandas as pd
import data_processing

stop_flag = False

def listen_for_stop():
    global stop_flag
    input("Pressione ENTER para parar...\n")
    stop_flag = True

threading.Thread(target=listen_for_stop, daemon=True).start()

file_path = r'supervised_output\Data_test.mat'
svm = joblib.load(r'supervised_output\svm_model.joblib')
scaler = joblib.load(r'unsupervised_output\scaler.joblib')

for window_30s in buffer.stream_30s_windows(file_path):
    if stop_flag:
        print("Execução parada pelo usuário.")
        break
    processed_dict = signal_processing.FilterSignals(window_30s, fs=1000)
    
    features = feature_extraction.featureVect(processed_dict)
    df_features = pd.DataFrame(features)
    Data, _ = data_processing.Standardization(df_features, scaler=scaler)
    
    prediction = svm.predict(Data)
    print(prediction)