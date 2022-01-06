import os
import numpy as np
import concurrent.futures
from itertools import repeat
from collections import deque
import threading
from tensorflow import keras
import multiprocessing
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K
import tensorflow as tf
import time
import random
import json
from typing import Union

from settings import *

def create_frames(data:np.ndarray):
  number_of_frames = data.size // frame_shift
  frames = [data[idx * frame_shift : FRAME_SIZE + idx * frame_shift] for idx in range(number_of_frames)]

  # Pad last frame if needed
  frames[len(frames) - 1] = np.pad(frames[len(frames) - 1], (0, FRAME_SIZE - frames[len(frames) - 1].shape[0]), "constant")
  return frames

def noise_data(data, fs):
  sdata = data.copy()

  length_in_secs = sdata.size / fs
  time = np.linspace(0, length_in_secs, sdata.size, endpoint=False)

  for _ in range(random.randint(1, MAXIMUM_NUMBER_OF_NOISE_CHANNELS)):
    sdata += (random.random() * NOISE_APLITUDE_MAX) * np.cos(2 * np.pi * (random.random() * NOISE_FREQ_MAX) * time + (random.random() * 2 * np.pi))
  
  return sdata

def find_files(files, dirs=[], extensions=[]):
    new_dirs = []
    for d in dirs:
        if not os.path.exists(d): continue
        
        try:
            new_dirs += [ os.path.join(d, f) for f in os.listdir(d) ]
        except OSError:
            if os.path.splitext(d)[1] in extensions:
                files.append(d)

    if new_dirs:
        find_files(files, new_dirs, extensions)
    else:
        return

def normalization(samples:np.ndarray):
    smpl = samples.copy()

    smpl = smpl - np.mean(smpl)

    max_abs_val = max(abs(smpl))

    # Normalization
    smpl /= max_abs_val
    return smpl

def convert_imag_to_parts(array:np.ndarray):
    tmp = np.empty((2, array.shape[0]))

    for idx, a in enumerate(array):
        tmp[0][idx] = np.abs(a)
        tmp[1][idx] = np.angle(a)

    return tmp

def convert_parts_to_complex(array:np.ndarray):
    tmp = np.empty((array.shape[1],))

    for idx in range(array.shape[0]):
        tmp[idx] = array[0][idx] * np.exp(1j*array[1][idx])

    return tmp

def data_generation(files, batch_size):
  try:
    # Initialization
    X = np.empty((batch_size, 2, FRAME_SIZE - FRAME_OVERLAP + 2))
    y = np.empty((batch_size, 2, FRAME_SIZE - FRAME_OVERLAP + 2))

        # Generate data
    for idx, file in enumerate(files):
        freq = float(file[:-4].split("_")[-1])

        try:
            loaded_data = np.load(file, allow_pickle=True)

            X[idx,] = np.pad(loaded_data[1], ((0, 0), (0, (FRAME_SIZE - FRAME_OVERLAP + 2) - loaded_data[1].shape[1])), "constant")
            y[idx,] = np.pad(loaded_data[0], ((0, 0), (0, (FRAME_SIZE - FRAME_OVERLAP + 2) - loaded_data[0].shape[1])), "constant")
        except Exception as e:
            print(f"[WARNING] Failed to create new batch\n{e}")
            return None

    return X, y
  except KeyboardInterrupt:
    return None

class DataGenerator(keras.utils.Sequence, threading.Thread):
    def __init__(self, path, batch_size=32, preload_size=64, workers:Union[None, int]=None, shuffle=True):
        super(DataGenerator, self).__init__()
        self.executor = concurrent.futures.ProcessPoolExecutor(multiprocessing.cpu_count() if workers is None else workers)
        self.__terminate = False;

        self.files = []
        self.preload_size = preload_size
        find_files(self.files, dirs=[path], extensions=[".npy"])

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.queue = deque()

        self.length = int(np.floor(len(self.files) / self.batch_size))
        self.index = 0
        self.daemon = True
        self.shuffle_data()
        self.start()

    def __del__(self):
        self.__terminate = True
        self.executor.shutdown(False, cancel_futures=True)

    def __len__(self):
        return self.length

    def run(self) -> None:
      try:
        while True:
          if self.__terminate: break

          if len(self.queue) < self.preload_size:
            batches = [self.get_batch_of_paths() for _ in range(self.preload_size)]
              
            return_data = self.executor.map(data_generation, batches, repeat(self.batch_size))

            for ret_dat in return_data:
              if ret_dat is not None:
                self.queue.append(ret_dat)
          else: time.sleep(0.1)
      except KeyboardInterrupt:
        self.executor.shutdown(False, cancel_futures=True)

    def get_batch_of_paths(self):
        if (self.index + 1) >= self.length:
            self.shuffle_data()
            self.index = 0

        files = self.files[self.index*self.batch_size:(self.index+1)*self.batch_size]
        self.index += 1
        return files

    def __getitem__(self, _):
        while len(self.queue) == 0: time.sleep(0.01)
        return self.queue.pop()

    def shuffle_data(self):
        if self.shuffle == True:
            np.random.shuffle(self.files)

    def on_epoch_end(self):
        pass

class BackupCallback(callbacks.Callback):
  def __init__(self, backup_path, best_weights_path=None, monitor_value="val_loss", monitor_value_mode="min", min_delta=0.0001):
    super(BackupCallback, self).__init__()

    self.backup_path = backup_path
    self.value_save_name = monitor_value
    self.value_mode = monitor_value_mode
    self.best_weights_path = best_weights_path
    self.min_delta = min_delta

    setattr(self, monitor_value, np.Inf if monitor_value_mode == "min" else -np.Inf)

    self.start_data = None
    if os.path.exists(backup_path):
      with open(backup_path, "r") as f:
        self.start_data = json.load(f)

    if self.start_data and monitor_value:
      try:
        setattr(self, monitor_value, self.start_data[monitor_value])
      except:
        pass

  def on_epoch_end(self, epoch, logs=None):
    self.save_backup(epoch, self.model, logs)

  def save_backup(self, epoch, model, logs):
    logs = logs or {}

    try:
      data = {
        "epoch" : epoch,
        "lr" : float(K.eval(model.optimizer.learning_rate))
      }

      try:
        if self.value_save_name is not None and self.value_mode is not None:
          value = logs.get(self.value_save_name)
          attr = float(getattr(self, self.value_save_name))

          if self.value_mode == "min":
            if (value + self.min_delta) < attr:
              setattr(self, self.value_save_name, value)
              attr = value

              print(f"\nNew record value of {self.value_save_name} reached {round(value, 4)}")

              if self.best_weights_path is not None:
                model.save_weights(self.best_weights_path)
            else:
              print(f"\nValue of {self.value_save_name} is {round(value, 4)} that is higher than current record of {round(attr, 4)}")
          elif self.value_mode == "max":
            if (value - self.min_delta) > attr:
              setattr(self, self.value_save_name, value)
              attr = value

              print(f"\nNew record value of {self.value_save_name} reached {round(value, 4)}")

              if self.best_weights_path is not None:
                model.save_weights(self.best_weights_path)
            else:
              print(f"\nValue of {self.value_save_name} is {round(value, 4)} that is lower than current record of {round(attr, 4)}")

          data[self.value_save_name] = attr
      except Exception as e:
        print(f"\n[ERROR] Some error when saving checkpoint record\n{e}")

      with open(self.backup_path, "w") as f:
        json.dump(data, f)
    except Exception as e:
      print(f"\n[WARNING] Failed to save backup\n{e}")

  def get_value(self):
    try:
      return getattr(self, self.value_save_name)
    except:
      return None

  def get_start_epoch(self):
    if self.start_data is not None:
      return self.start_data["epoch"]
    return 0

  def on_train_begin(self, logs=None):
    if self.start_data is not None:
      K.set_value(self.model.optimizer.learning_rate, self.start_data["lr"])

def noiseToSignalLoss(y_true, y_pred):
    losses = tf.math.divide(tf.math.reduce_sum(tf.math.pow(tf.math.abs(tf.math.subtract(y_true,y_pred)),2)),
                            tf.math.reduce_sum(tf.math.pow(tf.math.abs(y_true),2)))
    return tf.reduce_mean(losses)

def SNR(y_true, y_pred):
  return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
