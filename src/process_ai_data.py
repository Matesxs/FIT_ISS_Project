import random
import subprocess
import shutil
import gc
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
import math
from pydub import AudioSegment
import multiprocessing
from functools import partial
import os
import librosa
import librosa.display
import scipy
import matplotlib.pyplot as plt

from helpers import *

def convert_all_files():
    for file_path in file_paths:
        cleaned_name = file_path[:-4]

        if file_path[-4:] == ".mp3":
            new_name = cleaned_name + ".wav"

            sound = AudioSegment.from_file(file_path, format="mp3")
            sound.export(new_name, "wav")

            os.remove(file_path)
        elif file_path[-4:] == ".mp4":
            subprocess.Popen(f"ffmpeg -i {file_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {cleaned_name}.wav").wait()
            os.remove(file_path)

def process_one_file(smpls, fs):
  # Split and normalize
  normalized_samples = normalization(smpls)

  # Create noised data and normalize them
  noised_samples = noise_data(normalized_samples, fs)
  noised_samples = normalization(noised_samples)

  # Set signals to be between 0 and 1
  normalized_samples += 1
  noised_samples += 1
  normalized_samples = normalization(normalized_samples)
  noised_samples = normalization(noised_samples)

  return normalized_samples, noised_samples

def move_file(srcs, dest):
  try:
    for src in srcs:
      os.rename(src, os.path.join(dest, Path(src).name))
  except KeyboardInterrupt:
    pass

def move_files(file_paths, target_path):
  try:
    with multiprocessing.Pool(multiprocessing.cpu_count() * 2) as p:
      p.map(partial(move_file, dest=target_path), [file_paths[(i * 20) : ((i * 20) + 20)] for i in range(math.ceil(len(file_paths) / 20))])
  except KeyboardInterrupt:
    pass

if __name__ == "__main__":
  file_paths = []
  find_files(file_paths, dirs=[DATA_PATH], extensions=[".mp3", ".mp4"])
  print("Files to convert")
  print(len(file_paths))

  convert_all_files()
      
  already_used_filenames = []
  find_files(already_used_filenames, dirs=[TEST_DATASET_PATH, TRAIN_DATASET_PATH], extensions=[".npy"])

  if not os.path.exists("tmp_dataset"):
    os.mkdir("tmp_dataset")

    file_paths = []
    find_files(file_paths, dirs=[DATA_PATH], extensions=[".wav"])

    print("Numbe of source files")
    print(len(file_paths))

    window = scipy.signal.hamming(FRAME_SIZE, sym=False)

    for file_path in tqdm(file_paths):
      cleaned_name = Path(file_path).name[:-4]

      if (any([cleaned_name in file_name for file_name in already_used_filenames]) or
          any([cleaned_name in file_name for file_name in os.listdir("tmp_dataset")])):
        print(f"Skipping {cleaned_name}")
        continue

      smpls, f = librosa.load(file_path, sr=MAX_SAMPLE_FREQUENCY, mono=True)
      gc.collect()

      print(f"Processing {cleaned_name}")
          
      random.seed()
      norm_s, nois_s = process_one_file(smpls, f)

      data1 = librosa.stft(norm_s, n_fft=FRAME_SIZE, hop_length=FRAME_SIZE - FRAME_OVERLAP, window=window, center=True)
      data2 = librosa.stft(nois_s, n_fft=FRAME_SIZE, hop_length=FRAME_SIZE - FRAME_OVERLAP, window=window, center=True)

      # fig, ax = plt.subplots()
      # img = librosa.display.specshow(librosa.amplitude_to_db(abs(data1), ref=np.max), y_axis='log', x_axis='time', ax=ax)
      # ax.set_title('Power spectrogram')
      # fig.colorbar(img, ax=ax, format="%+2.0f dB")
      # plt.show()

      # fig, ax = plt.subplots()
      # img = librosa.display.specshow(librosa.amplitude_to_db(abs(data2), ref=np.max), y_axis='log', x_axis='time', ax=ax)
      # ax.set_title('Power spectrogram')
      # fig.colorbar(img, ax=ax, format="%+2.0f dB")
      # plt.show()

      for idx in range(data1.shape[1]):
        if not os.path.exists(f"tmp_dataset/{idx}_{cleaned_name}_{f}"):
          ds = convert_imag_to_parts(data1[:, idx])
          dn = convert_imag_to_parts(data2[:, idx])
          np.save(f"tmp_dataset/{idx}_{cleaned_name}_{f}", np.array([ds, dn]))

      gc.collect()

  file_paths = []
  find_files(file_paths, dirs=["tmp_dataset"], extensions=[".npy"])
  number_of_files = len(file_paths)

  print("Files to sort")
  print(number_of_files)

  if number_of_files > 0:
    random.shuffle(file_paths)

    if not os.path.exists(TEST_DATASET_PATH):
      os.mkdir(TEST_DATASET_PATH)

    if not os.path.exists(TRAIN_DATASET_PATH):
      os.mkdir(TRAIN_DATASET_PATH)

    print("Moving files")
    if VAL_SPLIT is not None:
      valid_file_count = int(number_of_files * VAL_SPLIT)
      move_files(file_paths[:valid_file_count], TEST_DATASET_PATH)
      move_files(file_paths[valid_file_count:], TRAIN_DATASET_PATH)
    else:
      move_files(file_paths, TRAIN_DATASET_PATH)

  shutil.rmtree("tmp_dataset")