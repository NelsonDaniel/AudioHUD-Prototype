import zerorpc
import os
import librosa
from utils import spectrum_fast
import pickle

class StreamingRPC(object):
    def get_input(self):
        file_path = os.path.join(os.path.dirname(__file__), "../sample-data/split0_ov1_0_A.wav")
        file, _ = librosa.load(file_path, sr=32000, mono=False)
        magnitude_spectrum = spectrum_fast(file, nperseg=512, noverlap=112, window="hamming", output_phase=False)
        serialized_bytes = pickle.dumps(magnitude_spectrum)
        return serialized_bytes

s = zerorpc.Server(StreamingRPC())
s.bind("tcp://0.0.0.0:4242")
s.run()
