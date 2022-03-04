import torch
from seldnet import Seldnet_augmented
from utils import load_model, spectrum_fast
import pickle
import numpy as np
from torchsummary import summary
import sys
import zerorpc
import os
import librosa
import pandas as pd

# Input Data
input_filename = 'split0_ov1_0_A.wav'
file_path = os.path.join(os.path.dirname(__file__), "../sample-data/{}".format(input_filename))
file, _ = librosa.load(file_path, sr=32000, mono=False)
magnitude_spectrum = spectrum_fast(file, nperseg=512, noverlap=112, window="hamming", output_phase=False)
model_input = torch.tensor(magnitude_spectrum).unsqueeze(0)
# Label Dict
int_to_label = {
    0:'Chink_and_clink',
    1:'Computer_keyboard',
    2:'Cupboard_open_or_close',
    3:'Drawer_open_or_close',
    4:'Female_speech_and_woman_speaking',
    5:'Finger_snapping',
    6:'Keys_jangling',
    7:'Knock',
    8:'Laughter',
    9:'Male_speech_and_man_speaking',
    10:'Printer',
    11:'Scissors',
    12:'Telephone',
    13:'Writing',
    14:'NOTHING'
    }

def get_pytorch_model(pretrained_path):
    # load model
    device = 'cpu'
    use_cuda = False
    model = Seldnet_augmented(
                time_dim=2400,
                freq_dim=256,
                input_channels=4,
                output_classes=14,
                pool_size=[[8, 2], [8, 2], [2, 2], [1, 1]],
                pool_time=True,
                rnn_size=256,
                n_rnn=3,
                fc_size=1024,
                dropout_perc=0.3,
                cnn_filters=[64, 128, 256, 512],
                class_overlaps=3,
                verbose=False,
            )

    model = model.to(device)
    #summary(model)
    load_model(model, None, pretrained_path, use_cuda)
    # set model to inference mode
    model.eval()
    return model

def rp(p):
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, p)

model = get_pytorch_model(rp("pretrained/seldnet.pth"))
with torch.no_grad():
    sed, doa = model(model_input)

sed = sed.cpu().numpy().squeeze()
doa = doa.cpu().numpy().squeeze()

def convert_raw_predictions(sed, doa, num_frames=300, num_classes=14, max_overlaps=3):
    output = {}
    for frame_num, (cls, label) in enumerate(zip(sed, doa)):  #iterate all time frames
        cls = np.round(cls)  # 0.5 threshold
        label = label * 2.0  # Multiple each raw location with 2.0 to restore original range of magnitures [-2,2]
        label = label.reshape(num_classes, max_overlaps, 3)  
        if np.sum(cls) == 0:  #if no sounds are detected in a frame output : [NOTHING, nan, nan, nan]
            output[frame_num] = [int_to_label[14], int_to_label[14], int_to_label[14], int_to_label[14]]
        else:
            for idx, val in enumerate(cls):  #iterate all events
                if val != 0:
                    predicted_class = int(idx / max_overlaps)
                    num_event = int(idx % max_overlaps)
                    temp_list = [int_to_label[int(predicted_class)], label[predicted_class][num_event][0], 
                    label[predicted_class][num_event][1], 
                    label[predicted_class][num_event][2]]

                    output[frame_num] = temp_list

    return output


# Will save output csv file in sample_data
prediction = convert_raw_predictions(sed, doa)
prediction = list(prediction.values())
df = pd.DataFrame(prediction, columns=['Class','X','Y', 'Z'])
df.index.name = 'Frame Number'
df.to_csv('../sample-data/'+'output_'+input_filename[:-4]+'.csv')
