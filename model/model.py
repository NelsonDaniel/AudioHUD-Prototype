from pyexpat import model
import torch
import zerorpc
from seldnet import Seldnet_augmented
import utils
import os
import pickle
import numpy as np

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
    13:'Writing'
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
    utils.load_model(model, None, pretrained_path, use_cuda)

    # set model to inference mode
    model.eval()
    return model

def rp(p):
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, p)

c = zerorpc.Client()
c.connect("tcp://127.0.0.1:4242")
serialized_input = c.get_input()
deserialized_input = pickle.loads(serialized_input)
model_input = torch.tensor(deserialized_input).float().unsqueeze(0)
model = get_pytorch_model(rp("pretrained/seldnet.pth"))
with torch.no_grad():
    sed, doa = model(model_input)

sed = sed.cpu().numpy().squeeze()
doa = doa.cpu().numpy().squeeze()
n = sed.shape[1]
x = doa[:, :n]
y = doa[:, n : n * 2]
z = doa[:, n * 2 :]
positions = np.arange(0, sed.shape[0] + 1, 50)
positions_set = set(positions)
labels = np.array(positions / 10, dtype="int32")

for i in range(sed.shape[0]):
    arg_max_label = np.argmax(sed[i])
    label_int = arg_max_label // 3
    label = int_to_label[label_int]
    result = [
        np.argmax(x[i][label_int]) ,
        np.argmax(y[i][label_int]),
        np.argmax(z[i][label_int]),
        label
    ]
    
    print(result)

