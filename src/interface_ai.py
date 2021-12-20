import matplotlib.pyplot as plt
import soundfile as sf
from scipy.io import wavfile
import numpy as np
import math
import os
from tensorflow.keras.models import load_model

from helpers import *

MODEL_PATH = "model.h5"

def denoise(sample, fs, model):
  number_of_frames = math.ceil(sample.size / FRAME_SIZE)
  frames = [samples_normal[idx * frame_shift : FRAME_SIZE + idx * frame_shift] for idx in range(number_of_frames)]

  # Pad last frame with zeroes
  frames[len(frames) - 1] = np.pad(frames[len(frames) - 1], (0, FRAME_SIZE - frames[len(frames) - 1].shape[0]), "constant")

  frames = np.array(frames)

  fft = np.array([convert_imag_to_parts(np.fft.fft(frame)) for frame in frames])

  output_frames = model.predict([frames, fft, np.ones((frames.shape[0], 1)) * fs])

  final_frame = None
    
  for frame in output_frames:
    frame = np.reshape(frame, (FRAME_SIZE,))

    if final_frame is None:
      final_frame = frame
    else:
      final_frame = np.concatenate([final_frame, frame], axis=0)

  if final_frame is not None: final_frame = normalization(final_frame)
  return final_frame

if __name__ == "__main__":
  if not os.path.exists(MODEL_PATH):
    print("Path to model not found")
    exit(-1)
    
  model = load_model(MODEL_PATH, custom_objects={"noiseToSignalLoss" : noiseToSignalLoss, "SNR" : SNR})

  samples_orig, sample_freq = sf.read("../audio/xdousa00.wav")

  samples_normal = normalization(samples_orig) + 1
  samples_normal = normalization(samples_normal)

  plt.figure(figsize=(18,8))
  plt.title("Normalizovaný vstupní signál")
  plt.plot(np.arange(samples_normal.size) / sample_freq, samples_normal)
  plt.gca().set_xlabel('$t[s]$')
  plt.gca().set_ylabel('$Amplituda[-]$')
  plt.show()

  cleared_signal = denoise(samples_normal, sample_freq, model)

  print(cleared_signal.shape)

  plt.figure(figsize=(18,8))
  plt.title("Vyčištěný signál")
  plt.plot(np.arange(cleared_signal.size) / sample_freq, cleared_signal)
  plt.gca().set_xlabel('$t[s]$')
  plt.gca().set_ylabel('$Amplituda[-]$')
  plt.show()

  wavfile.write("clean_test.wav", sample_freq, cleared_signal)