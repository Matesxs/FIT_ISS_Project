import matplotlib.pyplot as plt
import soundfile as sf
import scipy
from scipy.io import wavfile
import numpy as np
import librosa
import librosa.display
import os
from tensorflow.keras.models import load_model

from helpers import *

MODEL_PATH = "model.h5"

def denoise(sample, fs, model):
  window = scipy.signal.hamming(FRAME_SIZE, sym=False)
  data = librosa.stft(sample, n_fft=FRAME_SIZE, hop_length=FRAME_SIZE - FRAME_OVERLAP, window=window, center=True)
  print(data.shape)

  fig, ax = plt.subplots()
  img = librosa.display.specshow(librosa.amplitude_to_db(abs(data), ref=np.max), y_axis='log', x_axis='time', ax=ax)
  ax.set_title('Power spectrogram of input signal')
  fig.colorbar(img, ax=ax, format="%+2.0f dB")
  plt.show()

  batches = [convert_imag_to_parts(np.pad(data[:, idx], (0, (FRAME_SIZE - FRAME_OVERLAP + 2) - data[:, idx].shape[0]), "constant")) for idx in range(data.shape[1])]
  print(np.array(batches).shape)

  output_frames = model.predict(np.array(batches))

  final_frames = np.zeros((FRAME_SIZE - FRAME_OVERLAP + 1, output_frames.shape[0]))

  for idx, out_frame in enumerate(output_frames):
    final_frames[:-1, idx] = convert_parts_to_complex(out_frame)

  print(final_frames.shape)
  final_signal = librosa.istft(final_frames, window=window, hop_length=FRAME_SIZE - FRAME_OVERLAP, center=True)
  data2 = librosa.stft(final_signal, n_fft=FRAME_SIZE, hop_length=FRAME_SIZE - FRAME_OVERLAP, window=window, center=True)

  fig, ax = plt.subplots()
  img = librosa.display.specshow(librosa.amplitude_to_db(abs(data2), ref=np.max), y_axis='log', x_axis='time', ax=ax)
  ax.set_title('Power spectrogram of cleaned signal')
  fig.colorbar(img, ax=ax, format="%+2.0f dB")
  plt.show()

  return final_signal

if __name__ == "__main__":
  if not os.path.exists(MODEL_PATH):
    print("Path to model not found")
    exit(-1)
    
  model = load_model(MODEL_PATH, custom_objects={"noiseToSignalLoss" : noiseToSignalLoss, "SNR" : SNR})

  samples_orig, sample_freq = librosa.load("../audio/xdousa00.wav", sr=MAX_SAMPLE_FREQUENCY, mono=True)

  samples_normal = normalization(samples_orig) + 1
  samples_normal = normalization(samples_normal)

  plt.figure(figsize=(18,8))
  plt.title("Normalizovaný vstupní signál")
  plt.plot(np.arange(samples_normal.size) / sample_freq, samples_normal)
  plt.gca().set_xlabel('$t[s]$')
  plt.gca().set_ylabel('$Amplituda[-]$')
  plt.show()

  cleared_signal = denoise(samples_normal, sample_freq, model)

  plt.figure(figsize=(18,8))
  plt.title("Vyčištěný signál")
  plt.plot(np.arange(cleared_signal.size) / sample_freq, cleared_signal)
  plt.gca().set_xlabel('$t[s]$')
  plt.gca().set_ylabel('$Amplituda[-]$')
  plt.show()

  wavfile.write("clean_test.wav", sample_freq, (cleared_signal * np.iinfo(np.int16).max).astype(np.int16))