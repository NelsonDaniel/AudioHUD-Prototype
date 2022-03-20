# Imports
import os
import sys
import numpy as np
import librosa
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import zmq
import json

# socket
os.chdir(sys._MEIPASS)
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# CONSTANTS
CLS_TO_IDX = {'Chink_and_clink':0,
               'Computer_keyboard':1,
               'Cupboard_open_or_close':2,
               'Drawer_open_or_close':3,
               'Female_speech_and_woman_speaking':4,
               'Finger_snapping':5,
               'Keys_jangling':6,
               'Knock':7,
               'Laughter':8,
               'Male_speech_and_man_speaking':9,
               'Printer':10,
               'Scissors':11,
               'Telephone':12,
               'Writing':13,
                'NOTHING': 14}

IDX_TO_CLS = {v: k for k, v in CLS_TO_IDX.items()}

# audio file time
FILE_DURATION = 30
# audio sampling frequency
SAMPLING_FREQUENCY = 32000
# audio chunks duration
DURATION = 1.0
# Output generated every 100 ms
STEP = 0.1
# Value for normalization for tanh
MAX_LOC_VALUE = 2.
# Number of output frames
NUM_FRAMES = 10

SCRIPT_PATH = os.path.dirname(__file__)
AUDIO_PATH = os.path.join(SCRIPT_PATH,'static', 'prototype_test_audio.npy')
STATDICT_PATH = os.path.join(SCRIPT_PATH, 'static', 'stat_dict.npy')


def compute_mel_spectrogram(sound, nfft = 2048, hop_length = 400, n_mels = 128, top_db = 80, sr = SAMPLING_FREQUENCY,
                          cut_dc=True, output_phase=True, cut_last_timeframe=True):
    '''
    Compute Non Linear Spectogram in dB scale
    '''
    
    S = librosa.feature.melspectrogram(sound, sr=sr, n_fft=nfft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, top_db = top_db)
    if cut_last_timeframe:
        output = S_DB[:,:,:-1]
    return output
    
def load_stat_dict():
    stat_dict = np.load(STATDICT_PATH, allow_pickle = True).item()
    mean_ = stat_dict['mean']
    std_ = stat_dict['std']
    return mean_, std_

def process_data(stereo):
    '''
    Input of 2 x 32000 (1 second audio)
    '''
    spec = compute_mel_spectrogram(stereo)
    spec = np.expand_dims(spec, axis = 0)
    mean_, std_ = load_stat_dict()
    spec_norm = (spec - mean_) / std_
    return spec_norm

if __name__ == "__main__":
    '''
    Accept 1 second of audio -> process audio -> generate model inference
    '''
    prototype_test_audio = np.load(AUDIO_PATH)
    for i in range(len(prototype_test_audio)):
        print("running")
        message = socket.recv()
        spec_norm = process_data(prototype_test_audio[i])
        socket.send_string(json.dumps(str(spec_norm)))
