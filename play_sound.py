import sounddevice as sd
import soundfile as sf
import multiprocessing
import os

# Global variables to track current playing process and prediction
current_process = None
current_prediction = None

def _play_loop(wav_path):
    data, fs = sf.read(wav_path, dtype='float32')
    sd.play(data, fs)
    sd.wait()

def _get_sound_path(prediction):
    sounds_dir = 'sounds'
    sound_map = {
        0: 'Wake.wav',
        1: 'NREM.wav',
        2: 'REM.wav'
    }
    filename = sound_map.get(prediction)
    if filename:
        return os.path.join(sounds_dir, filename)
    return None

def handle_prediction(prediction):

    global current_process, current_prediction
    print(f"\nhandle_prediction called with {prediction}")

    path = _get_sound_path(prediction)

    if prediction == current_prediction:
        print(f"Sound path: {path}")
        return

    current_prediction = prediction

    if current_process and current_process.is_alive():
        print("Terminating previous sound process")
        current_process.terminate()
        current_process.join()

    path = _get_sound_path(prediction)
    print(f"Sound path: {path}")
    if path and os.path.exists(path):
        current_process = multiprocessing.Process(target=_play_loop, args=(path,))
        current_process.start()
    else:
        print(f"Warning: Audio file not found for prediction {prediction}: {path}")

def stop_audio():
    """
    Stops any currently playing audio.
    """
    global current_process
    if current_process and current_process.is_alive():
        current_process.terminate()
        current_process.join()
        current_process = None
