import cv2, imutils
import numpy as np
import time
import base64
import threading
import wave
import pyaudio
import pickle
import struct
import sys
import queue
import os
# Imports
import librosa
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn as nn
import torch.utils.data as utils
import zmq

# MODEL INFERENCE
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

WORKING_DIR = os.getcwd()
MODEL_PATH = os.path.join(WORKING_DIR, 'MODEL', 'prototype_model_v1')

class baseline_model(nn.Module):
    def __init__(self, time_dim = 80, freq_dim=128, input_channels=2, output_classes=15,
                 pool_size=[[8,2],[8,2],[2,2],[1,1]], cnn_filters=[32,64,128,256], pool_time=True,
                 rnn_size=128, n_rnn=3, fc_size=512, dropout_perc=0.3,
                 verbose=False):
        super(baseline_model, self).__init__()
        self.verbose = verbose
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        sed_output_size = output_classes    #here 3 is the max number of simultaneus sounds from the same class
        doa_output_size = sed_output_size * 3   #here 3 is the number of spatial dimensions xyz
        if pool_time:
            self.time_pooled_size = int(time_dim/ np.prod(pool_size, axis=0)[-1])
        else:
            self.time_pooled_size = time_dim
        #building CNN feature extractor
        conv_layers = []
        in_chans = input_channels
        for i, (p,c) in enumerate(zip(pool_size, cnn_filters)):
            curr_chans = c

            if pool_time:
                pool = [p[0],p[1]]
            else:
                pool = [p[0],1]
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_chans, out_channels=curr_chans,
                                kernel_size=3, stride=1, padding=1),  #padding 1 = same with kernel = 3
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                    nn.MaxPool2d(pool),
                    nn.Dropout(dropout_perc)))
            in_chans = curr_chans

        self.cnn = nn.Sequential(*conv_layers)

        self.rnn = nn.GRU(256, rnn_size, num_layers=n_rnn, batch_first=True,
                          bidirectional=True, dropout=dropout_perc)

        self.sed = nn.Sequential(
                    nn.Linear(rnn_size*2, fc_size),
                    nn.ReLU(),
                    nn.Linear(fc_size, fc_size),
                    nn.ReLU(),
                    nn.Linear(fc_size, fc_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_perc),
                    nn.Linear(fc_size, sed_output_size),
                    nn.Sigmoid())

        self.doa = nn.Sequential(
                    nn.Linear(rnn_size*2, fc_size),
                    nn.ReLU(),
                    nn.Linear(fc_size, fc_size),
                    nn.ReLU(),
                    nn.Linear(fc_size, fc_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_perc),
                    nn.Linear(fc_size, doa_output_size),
                    nn.Tanh())

    def forward(self, x):
        x = self.cnn(x)
        if self.verbose:
            print ('cnn out ', x.shape)    #target dim: [batch, n_cnn_filters, 2, time_frames]
        x = x.permute(0,3,1,2) #[batch, time, channels, freq]
        if self.verbose:
            print ('permuted: ', x.shape)    #target dim: [batch, time_frames, n_cnn_filters, 2]
        x = x.reshape(x.shape[0], self.time_pooled_size, -1)
        if self.verbose:
            print ('reshaped: ', x.shape)    #target dim: [batch, 2*n_cnn_filters]
        x, h = self.rnn(x)
        if self.verbose:
            print ('rnn out:  ', x.shape)    #target dim: [batch, 2*n_cnn_filters]
        sed = self.sed(x)
        doa = self.doa(x)
        if self.verbose:
            print ('sed prediction:  ', sed.shape)  #target dim: [batch, time, sed_output_size]
            print ('doa prediction: ', doa.shape)  #target dim: [batch, time, doa_output_size]

        return sed, doa

def extract_audio_chunks(audio_filename):   
    prototype_audio_array = []
    for i in range(int(FILE_DURATION * 2)):
        temp_sample, sr = librosa.load(audio_filename, sr = SAMPLING_FREQUENCY, 
                                    duration = DURATION, offset = i, mono = False)
        prototype_audio_array.append(temp_sample)
    prototype_audio_array = np.array(prototype_audio_array)
    return prototype_audio_array
    
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
    stat_dict = np.load('stat_dict.npy', allow_pickle = True).item()
    mean_ = stat_dict['mean']
    std_ = stat_dict['std']
    return mean_, std_

def process_data(stereo, mean_, std_):
    '''
    Input of 2 x 32000 (1 second audio)
    '''
    spec = compute_mel_spectrogram(stereo)
    spec = np.expand_dims(spec, axis = 0)
    spec_norm = (spec - mean_) / std_
    spec_tensor = torch.tensor(spec_norm).float()
    return spec_tensor

def load_model(model, optimizer, path, cuda):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # load state dict of wrapped module
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    try:
        #model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(torch.load(checkpoint['model_state_dict'],
                                    map_location=lambda storage, location: storage),
                                    strict=False)
    except:
        # work-around for loading checkpoints where DataParallel was saved instead of inner module
        from collections import OrderedDict
        model_state_dict_fixed = OrderedDict()
        prefix = 'module.'
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith(prefix):
                k = k[len(prefix):]
            model_state_dict_fixed[k] = v
        model.load_state_dict(model_state_dict_fixed)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'state' in checkpoint:
        state = checkpoint['state']
    else:
        # older checkpoints only store step, rest of state won't be there
        state = {'step': checkpoint['step']}
    return state

def generate_inference_list(sed, doa):
    predictions = []
    for i, (c, l) in enumerate(zip(sed, doa)):
        c = np.round(c)
        l = l * MAX_LOC_VALUE
        l = l.reshape(15, 3)
        if np.sum(c) == 0:
            predictions.append([IDX_TO_CLS[14], 0, 0, 0])
        else:
            predicted_class = int(np.argmax(c))
            curr_list = [IDX_TO_CLS[int(predicted_class)], l[predicted_class][0], 
                         l[predicted_class][1], l[predicted_class][2]]
            predictions.append(curr_list)
    
    return predictions

def call(stereo):
    spec_tensor = process_data(stereo, mean_, std_)
    spec_tensor = spec_tensor.to(device)
    sed, doa = model(spec_tensor)
    sed = sed.detach().cpu().numpy().squeeze()
    doa = doa.detach().cpu().numpy().squeeze()
    predictions = generate_inference_list(sed, doa)
    return predictions
    
# Read Video
video_filename =  os.path.join('uploads', 'office-space-with-audio.mp4')
audio_filename = 'temp_audio.wav'
# Extract and Preprocess Audio File from Video
cmd = "ffmpeg -i {} -q:a 0 -map a -ac 2 -ar 32000 -y {}".format(video_filename, audio_filename)
os.system(cmd)
# Read Audio File to Array in Chunks of 1 second
prototype_audio_array = extract_audio_chunks(audio_filename)
# Create Video Object and Get Video Attributes
vid = cv2.VideoCapture(video_filename)
FPS = vid.get(cv2.CAP_PROP_FPS)
TS = (1.2/FPS)
BREAK=False
totalNoFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
durationInSeconds = float(totalNoFrames) / float(FPS)
d=vid.get(cv2.CAP_PROP_POS_MSEC)

# Initialize Model
mean_, std_ = load_stat_dict()
if torch.cuda.is_available():
    device=torch.device('cuda:0')
else:
    device=torch.device('cpu')
model = baseline_model()
model.to(device)
state = load_model(model, None, MODEL_PATH, cuda = True)
model.eval()
# Make Queue for Buffering
q = queue.Queue(maxsize=10)

def video_stream_gen():
    WIDTH=1400
    while(vid.isOpened()):
        try:
            _,frame = vid.read()
            frame = imutils.resize(frame,width=WIDTH)
            q.put(frame)
        except:
            os._exit(1)
    BREAK=True
    vid.release()

def video_stream():
    global TS
    fps,st,frames_to_count,cnt = (0,0,20,0)
    cv2.namedWindow('PLAYING PROTOTYPE VIDEO')        
    cv2.moveWindow('PLAYING PROTOTYPE VIDEO', 5,10) 

    while True:
        frame = q.get()
        frame = cv2.putText(frame,'FPS: '+str(round(fps,1)),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
        if cnt == frames_to_count:
            try:
                fps = (frames_to_count/(time.time()-st))
                st=time.time()
                cnt=0
                if fps > FPS:
                    TS += 0.001
                elif fps < FPS:
                    TS -= 0.001
                else:
                    pass
            except:
                pass
        cnt += 1

        cv2.imshow('PLAYING PROTOTYPE VIDEO', frame)
        key = cv2.waitKey(int(1000*TS)) & 0xFF
        if key == ord('q'):
            os._exit(1)
            TS = False
            break	

def audio_stream():
    wf = wave.open("temp_audio.wav", 'rb')
    p = pyaudio.PyAudio()
    CHUNK = 32000
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    frames_per_buffer=CHUNK)
    while True:
        try:
            data = wf.readframes(CHUNK)
            stream.write(data)
        except:
            break

def run_model():
    for i in range(len(prototype_audio_array)):
        stereo = prototype_audio_array[i]
        pred = call(stereo)
        print(pred)
        time.sleep(1)


from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.submit(audio_stream)
    executor.submit(video_stream_gen)
    executor.submit(video_stream)
    executor.submit(run_model)
    