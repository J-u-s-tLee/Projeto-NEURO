import buffer
files = [r'Continuous\Mouse1\continuous1.mat']

for window_30s in buffer.stream_30s_windows(files):
    print(window_30s)
