import torch
import numpy as np
from scipy.signal import stft

def load_model(model, optimizer, path, cuda):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # load state dict of wrapped module
    if cuda:
        checkpoint = torch.load(path, map_location='cuda:0')
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

def spectrum_fast(x, nperseg=512, noverlap=128, window='hamming', cut_dc=True,
                  output_phase=True, cut_last_timeframe=True):
    '''
    Compute magnitude spectra from monophonic signal
    '''

    f, t, seg_stft = stft(x,
                        window=window,
                        nperseg=nperseg,
                        noverlap=noverlap)

    #seg_stft = librosa.stft(x, n_fft=nparseg, hop_length=noverlap)

    output = np.abs(seg_stft)

    if output_phase:
        phase = np.angle(seg_stft)
        output = np.concatenate((output,phase), axis=-3)

    if cut_dc:
        output = output[:,1:,:]

    if cut_last_timeframe:
        output = output[:,:,:-1]

    #return np.rot90(np.abs(seg_stft))
    return output

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

    