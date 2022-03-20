# Imports
import os
import sys
import numpy as np
import librosa
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn as nn
import torch.utils.data as utils
import zmq
import time
import json

# socket
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
MODEL_PATH = os.path.join(SCRIPT_PATH, 'static', 'prototype_model_v1')
AUDIO_PATH = os.path.join(SCRIPT_PATH,'static','prototype_test_audio.npy')
STATDICT_PATH = os.path.join(SCRIPT_PATH, 'static','stat_dict.npy')

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
    spec_tensor = torch.tensor(spec_norm).float()
    fake_target = np.zeros((1, 10, 60))
    fake_target_tensor = torch.tensor(fake_target).float()
    spec_dataset = utils.TensorDataset(spec_tensor, fake_target_tensor)
    spec_data = utils.DataLoader(spec_dataset, 1, shuffle = False)
    return spec_data

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

def calculate_location_sensitive_metrics(predictions, targets):
    TP = 0   #true positives
    FP = 0   #false positives
    FN = 0   #false negatives
    frames = {}
    for i in range(NUM_FRAMES):
        frames[i] = {'p':[], 't':[]}
    for i in predictions:
        frames[i[0]]['p'].append(i)
    for i in targets:
        frames[i[0]]['t'].append(i)

    for frame in range(NUM_FRAMES):
        t = frames[frame]['t']  #all true events for frame i
        p = frames[frame]['p']  #all predicted events for frame i

        num_true_items = len(t)
        num_pred_items = len(p)
        matched = 0
        match_ids = []       #all pred ids that matched
        match_ids_t = []     #all truth ids that matched

        if num_true_items == 0:         #if there are PREDICTED but not TRUE events
            FP += num_pred_items        #all predicted are false positive
        elif num_pred_items == 0:       #if there are TRUE but not PREDICTED events
            FN += num_true_items        #all predicted are false negative
        elif num_true_items == 0 and num_pred_items == 0:
            pass
        else:
            for i_t in range(len(t)):           #iterate all true events
                match = False       #flag for matching events
                #count if in each true event there is or not a matching predicted event
                true_class = t[i_t][1]          #true class
                true_coord = t[i_t][-3:]        #true coordinates
                for i_p in range(len(p)):       #compare each true event with all predicted events
                    pred_class = p[i_p][1]      #predicted class
                    pred_coord = p[i_p][-3:]    #predicted coordinates
                    spat_error = np.linalg.norm(true_coord-pred_coord)  #cartesian distance between spatial coords
                    if true_class == pred_class and spat_error < SPATIAL_THRESHOLD:  #if predicton is correct (same label + not exceeding spatial error threshold)
                        match_ids.append(i_p)   #append to pred matched ids
                        match_ids_t.append(i_t) #append to truth matched ids

            unique_ids = np.unique(match_ids)  #remove duplicates from matches ids lists
            unique_ids_t = np.unique(match_ids_t)
            matched = min(len(unique_ids), len(unique_ids_t))   #compute the number of actual matches without duplicates

            fn =  num_true_items - matched
            fp = num_pred_items - matched

            #add to counts
            TP += matched          #number of matches are directly true positives
            FN += fn
            FP += fp

    precision = TP / (TP + FP + sys.float_info.epsilon)
    recall = TP / (TP + FN + sys.float_info.epsilon)
    F_score = 2 * ((precision * recall) / (precision + recall + sys.float_info.epsilon))

    results = {'precision': precision,
               'recall': recall,
               'F score': F_score
               }
    
    return TP, FP, FN, F_score

def call(stereo):
    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    else:
        device=torch.device('cpu')
    spec_data = process_data(stereo)
    model = baseline_model()
    model.to(device)
    state = load_model(model, None, MODEL_PATH, cuda = True)
    model.eval()
    for example_num, (x, target) in enumerate(spec_data):
        x = x.to(device)
        sed, doa = model(x)
        sed = sed.detach().cpu().numpy().squeeze()
        doa = doa.detach().cpu().numpy().squeeze()
        predictions = generate_inference_list(sed, doa)
    return np.array(predictions)

if __name__ == "__main__":
    '''
    Accept 1 second of audio -> process audio -> generate model inference
    '''
    prototype_test_audio = np.load(AUDIO_PATH)
    for i in range(len(prototype_test_audio)):
        message = socket.recv()
        time.sleep(1)
        stereo = prototype_test_audio[i]
        pred = call(stereo)
        socket.send_string(json.dumps(str(pred[0])))
