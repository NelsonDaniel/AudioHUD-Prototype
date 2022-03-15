from msilib.schema import Error
import numpy as np
import pyaudio
import wave
import librosa
import io
from scipy.signal import stft

# Note: python 3.7 needs to used with this wheel of pyaudio: https://github.com/intxcc/pyaudio_portaudio
# Currently only works on Windows

p = pyaudio.PyAudio()  # Create an interface to PortAudio

sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 44100  # Record at 44100 samples per second
chunk = 4410 # Record in chunks of 1024 samples; Use 4410 because fs/chunk = 10 (10 frames a seconds)


# Select Device
# Returns a valid device ID 
# Need to verify deviceID is valid
def getAudioDevice():

    if p.get_default_host_api_info()['name'] == 'Windows WASAPI':
        return p.get_default_output_device_info()

    default_device_name = p.get_default_output_device_info()['name']
    for i in range(0, p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if p.get_host_api_info_by_index(info["hostApi"])["name"] == 'Windows WASAPI':
            if p.get_default_output_device_info()['name'] in info['name']:
                return info
    
    raise ValueError("Error selecting audio device.") 



# Returns a BytesIO object that stored all the frames
def save_data(frames):
    audioData = io.BytesIO() # Acts like a file, but in memory
    wf = wave.open(audioData, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    audioData.seek(0)
    return audioData

def spectrum_fast(x, nperseg=512, noverlap=384, window='hamming', cut_dc=True,
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
    


device_info = getAudioDevice()
device_id = device_info["index"]

channels = device_info["maxInputChannels"] if (device_info["maxOutputChannels"] < device_info["maxInputChannels"]) else device_info["maxOutputChannels"]
sampleRate = int(device_info["defaultSampleRate"])

# Create stream
# https://people.csail.mit.edu/hubert/pyaudio/docs/#pyaudio.Stream.__init__
stream = p.open(format=sample_format,
                channels=channels,
                rate=sampleRate,
                input=True,
                frames_per_buffer=chunk,
                input_device_index=device_info["index"],
                as_loopback=True
                )


seconds = 5 # Number of seconds you would like to capture audio
print('\nRecording from', device_info['name'], 'for ', seconds, 'seconds...\n')

frames = []  # Initialize array to store frames
num_frames = 0
max_frames = 10

while seconds>0:
    data = stream.read(chunk)   
    frames.append(data)
    num_frames += 1

    if num_frames == max_frames:
        wav_data = save_data(frames)

        x_byteio, sr = librosa.load(wav_data, sr=32000, mono=False)
        spf = spectrum_fast(x_byteio)
        print(spf.shape)
        # print("\n=====\n")
        # print(x_byteio, '\n')

        frames = []
        num_frames = 0
        seconds -= 1

# Maybe need to handle last chunk/partial chunk?
# For now, Cmd+C to stop program


# Stop and close the stream 
stream.stop_stream()
stream.close()

# Terminate the PortAudio interface
p.terminate()
print('Finished')

# # If you want to save to wav file - 
# filename = "output.wav"
# wf1 = wave.open(filename, 'wb')
# wf1.setnchannels(channels)
# wf1.setsampwidth(p.get_sample_size(sample_format))
# wf1.setframerate(fs)
# wf1.writeframes(b''.join(frames))
# wf1.close()
# x_wav, sr = librosa.load(filename, sr=32000, mono=False)





