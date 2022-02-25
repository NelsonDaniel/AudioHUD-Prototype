from SELDNET import Seldnet_augmented
import torch
import utils
import librosa
import torch.onnx 
import onnx
import onnxruntime
import numpy as np
import os

def rp(p):
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, p)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def convert_ONNX(model):
    onnx_model_path = rp("onnx_model\l3das.onnx") 
    dummy_file = rp("sample_data\split0_ov1_0_A.wav") 
    test_file = rp("sample_data\split0_ov1_1_A.wav")

    dummy, _ = librosa.load(dummy_file, sr=32000, mono=False)
    dummy = utils.spectrum_fast(dummy, nperseg=512, noverlap=112, window="hamming", output_phase=False)
    dummy = torch.tensor(dummy).float().unsqueeze(0)

    torch.onnx.export(model,         # model being run 
            dummy,       # model input (or a tuple for multiple inputs) 
            onnx_model_path,       # where to save the model  
            export_params=True,  # store the trained parameter weights inside the model file 
            opset_version=10,    # the ONNX version to export the model to 
            do_constant_folding=True,  # whether to execute constant folding for optimization 
            input_names = ['input'],   # the model's input names 
            output_names = ['output'], # the model's output names 
            dynamic_axes=  {'input' : {0 : 'batch_size'},    # variable length axes 
                            'output' : {0 : 'batch_size'}}) 

    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # get pytorch output
    x, _ = librosa.load(test_file, sr=32000, mono=False)
    x = utils.spectrum_fast(x, nperseg=512, noverlap=112, window="hamming", output_phase=False)
    x = torch.tensor(x).float().unsqueeze(0)
    with torch.no_grad():
        sed, doa = model(x)
    sed = sed.cpu().numpy().squeeze()
    doa = doa.cpu().numpy().squeeze()

    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs_sed, ort_out_doa = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(sed, ort_outs_sed[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(doa, ort_out_doa[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

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


pretrained_path = rp(f"checkpoint\seldnet.pth") 
model = get_pytorch_model(pretrained_path)
convert_ONNX(model)
