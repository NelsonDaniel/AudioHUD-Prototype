from SELDNET import Seldnet_augmented
import torch
import utils

# load model
pretrained_path = "model/pretrained/checkpoint.pt"
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
model.eval()
print("success")
