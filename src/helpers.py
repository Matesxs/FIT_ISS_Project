import os
import numpy as np
import concurrent.futures
from itertools import repeat
from collections import deque
import threading
from tensorflow import keras
from tensorflow.keras import layers, optimizers, Model, callbacks, initializers
import tensorflow.keras.backend as K
import tensorflow as tf
import time
import random
import json

from settings import *

def create_frames(data:np.ndarray):
  number_of_frames = data.size // frame_shift
  frames = [data[idx * frame_shift : FRAME_SIZE + idx * frame_shift] for idx in range(number_of_frames)]
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
    max_abs_val = max(abs(smpl))

    # Normalization
    smpl /= max_abs_val
    return smpl

def convert_imag_to_parts(array:np.ndarray):
    tmp = np.empty((array.shape[0], 2))

    for idx, a in enumerate(array):
        tmp[idx][0] = a.real
        tmp[idx][1] = a.imag

    return np.nan_to_num(tmp)

def data_generation(files, batch_size, dim):
  try:
    # Initialization
    X = np.empty((batch_size, *dim))
    fft = np.empty((batch_size, dim[0] // 2, 2))
    f = np.empty((batch_size), dtype=float)
    y = np.empty((batch_size, *dim))

        # Generate data
    for idx, file in enumerate(files):
        freq = float(file[:-4].split("_")[-1])

        try:
            loaded_data = np.load(file)

            X[idx,] = loaded_data[1].real
            fft[idx,] = convert_imag_to_parts(loaded_data[2])
            f[idx] = freq / MAX_SAMPLE_FREQUENCY
            y[idx,] = loaded_data[0].real
        except Exception as e:
            return None

    return [X, fft, f], y
  except KeyboardInterrupt:
    return None

class DataGenerator(keras.utils.Sequence, threading.Thread):
    def __init__(self, path, dim, batch_size=32, shuffle=True):
        super(DataGenerator, self).__init__()
        self.executor = concurrent.futures.ProcessPoolExecutor(8)
        self.__terminate = False;

        self.dim = dim
        self.files = []
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

          if len(self.queue) < 20:
            batches = [self.get_batch_of_paths() for _ in range(20)]
              
            return_data = self.executor.map(data_generation, batches, repeat(self.batch_size), repeat(self.dim))

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

  def get_value(self, value_name):
    try:
      return getattr(self, value_name)
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

def create_model(default_filter_size, number_of_layers, double_layers=False, lr=1e-4):
  inp1 = layers.Input(shape=(FRAME_SIZE,1), name="frame_input")
  inp2 = layers.Input(shape=(FRAME_SIZE // 2,2), name="fft_input")
  inp3 = layers.Input(shape=(1,), name="frequency_input")

  x = layers.Concatenate(name="data_conc")([inp1, inp2])
  x = layers.Flatten()(x)

  y = layers.Dense(64, name="freq_dense")(inp3)
  y = layers.LeakyReLU(0.2, name="freq_activation")(y)
  y = layers.Dropout(0.1, name="freq_dropout")(y)

  x = layers.Concatenate(name="freq_data_conc")([x, y])

  first_stage_layers = []
  for i in range(number_of_layers):
    x = layers.Dense(default_filter_size, kernel_initializer=initializers.RandomNormal(stddev=0.02), name=f"down_dense_{i}")(x)
    x = layers.LeakyReLU(0.2, name=f"down_dense_activation_{i}")(x)
    x = layers.BatchNormalization(axis=1, momentum=0.6, name=f"down_dense_batchnorm_{i}")(x)

    if i == 0:
      x = layers.Dropout(0.1, name=f"down_dense_dropout_{i}")(x)
    first_stage_layers.append(x)

    if double_layers:
      x = layers.Dense(default_filter_size, kernel_initializer=initializers.RandomNormal(stddev=0.02), name=f"down_dense_{i}d")(x)
      x = layers.LeakyReLU(0.2, name=f"down_dense_activation_{i}d")(x)
      x = layers.BatchNormalization(axis=1, momentum=0.6, name=f"down_dense_batchnorm_{i}d")(x)

    default_filter_size /= 2

  for i, layer in enumerate(reversed(first_stage_layers)):
    default_filter_size *= 2
    
    x = layers.Dense(default_filter_size, kernel_initializer=initializers.RandomNormal(stddev=0.02), name=f"up_dense_{i}")(x)
    x = layers.LeakyReLU(0.2, name=f"up_dense_activation_{i}")(x)
    x = layers.BatchNormalization(axis=1, momentum=0.6, name=f"up_dense_batchnorm_{i}")(x)

    x = layers.Add(name=f"skip_add_{i}")([x, layer])

  x = layers.Dense(FRAME_SIZE, name="output")(x)

  model = Model([inp1, inp2, inp3], x)
  
  model.compile(optimizer=optimizers.Adam(lr, 0.5), loss=noiseToSignalLoss, metrics=["mse", SNR])
  # model.compile(optimizer=optimizers.Adam(lr), loss="mse", metrics=[SNR])

  return model