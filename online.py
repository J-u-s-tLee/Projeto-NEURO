if __name__ == '__main__':
    import buffer
    import threading
    import joblib
    import signal_processing
    import feature_extraction
    import pandas as pd
    import data_processing
    import play_sound

    stop_flag = False

    def listen_for_stop():
        global stop_flag
        input("Press ENTER to STOP...\n")
        stop_flag = True

    threading.Thread(target=listen_for_stop, daemon=True).start()

    file_path = r'supervised_output\Data_test.mat'
    svm = joblib.load(r'supervised_output\svm_model.joblib')
    scaler = joblib.load(r'unsupervised_output\scaler.joblib')

    try:
        for window_30s in buffer.stream_30s_windows(file_path):
            if stop_flag:
                print("Execution stopped by user.")
                play_sound.stop_audio()
                break
            
            processed_dict = signal_processing.FilterSignals(window_30s, fs=1000)
            features = feature_extraction.featureVect(processed_dict)
            df_features = pd.DataFrame(features)
            Data, _ = data_processing.Standardization(df_features, scaler=scaler)
            
            prediction = int(svm.predict(Data)[0])  # get scalar int
            
            if prediction == 0:
                print('\nSleep Stage: Wake')
            elif prediction == 1:
                print('\nSleep Stage: NREM')
            elif prediction == 2:
                print('\nSleep Stage: REM')

            play_sound.handle_prediction(prediction)
            
    except Exception as e:
        print(f"Error during processing: {e}")
        play_sound.stop_audio()
