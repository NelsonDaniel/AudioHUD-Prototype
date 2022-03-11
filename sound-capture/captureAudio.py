import numpy
import pyaudio
import wave
import soundfile as sf
import librosa
from matplotlib import pylab
import io

# Note: python 3.7 needs to used with this wheel of pyaudio: https://github.com/intxcc/pyaudio_portaudio

p = pyaudio.PyAudio()  # Create an interface to PortAudio

sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 44100  # Record at 44100 samples per second
chunk = 4410 # Record in chunks of 1024 samples; Use 4410 because fs/chunk = 10 (10 frames a seconds)


# Select Device
# Returns a valid device ID 
# Need to verify deviceID is valid
def getAudioDevice():
    print ( "Available devices:\n")
    for i in range(0, p.get_device_count()):
        info = p.get_device_info_by_index(i) 
        print ( str(info["index"]) +  ": \t %s \n \t %s \n" % (info["name"], p.get_host_api_info_by_index(info["hostApi"])["name"]))
        pass
    return int(input("Enter device ID: "))


# Returns a BytesIO object to store audio data
def save_data():
    audioData = io.BytesIO() # Acts like a file, but in memory
    wf = wave.open(audioData, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    audioData.seek(0)
    return audioData
    

device_id = getAudioDevice()

device_info = p.get_device_info_by_index(device_id)
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


print('\nRecording', device_id, '...\n')

frames = []  # Initialize array to store frames
num_frames = 0
max_frames = 10

while 1:
    data = stream.read(chunk)   
    frames.append(data)
    num_frames += 1

    if num_frames == max_frames:
        wav_data = save_data()

        x_byteio, sr = librosa.load(wav_data, sr=32000, mono=False)
        print(x_byteio, '\n')

        frames = []
        num_frames = 0

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





