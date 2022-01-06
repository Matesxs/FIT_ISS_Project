from tensorflow.keras import layers, optimizers, Model
from tensorflow import keras

from settings import *

def create_model(l2_strength=0, lr=1e-4):
  inp1 = layers.Input(shape=(2, FRAME_SIZE - FRAME_OVERLAP + 2), name="frame_input")
  x = inp1

  x = layers.Reshape((2, FRAME_SIZE - FRAME_OVERLAP + 2, 1))(x)

  x1 = layers.Conv2D(filters=32, kernel_size=(2, 8), strides=1, padding="same", use_bias=False, kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = layers.LeakyReLU()(x1)
  x = layers.BatchNormalization()(x)

  x = layers.Conv2D(filters=32, kernel_size=(2, 5), strides=1, padding="same", use_bias=False, kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = layers.LeakyReLU()(x)
  x = layers.BatchNormalization()(x)

  x2 = layers.Conv2D(filters=64, kernel_size=(2, 5), strides=1, padding="same", use_bias=False, kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = layers.LeakyReLU()(x2)
  x = layers.BatchNormalization()(x)

  x = layers.Conv2D(filters=64, kernel_size=(2, 5), strides=(1, 2), padding="same", use_bias=False, kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = layers.LeakyReLU()(x)
  x = layers.BatchNormalization()(x)

  x = layers.Conv2D(filters=128, kernel_size=(2, 5), strides=1, padding="same", use_bias=False, kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = layers.LeakyReLU()(x)
  x = layers.BatchNormalization()(x)

  x = layers.Conv2D(filters=64, kernel_size=(2, 5), strides=1, padding="same", use_bias=False, kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = layers.UpSampling2D((1, 2))(x)
  x = layers.Add()([x, x2])
  x = layers.LeakyReLU()(x)
  x = layers.BatchNormalization()(x)

  x = layers.Conv2D(filters=64, kernel_size=(2, 5), strides=1, padding="same", use_bias=False, kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = layers.LeakyReLU()(x)
  x = layers.BatchNormalization()(x)

  x = layers.Conv2D(filters=32, kernel_size=(2, 5), strides=1, padding="same", use_bias=False, kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = layers.Add()([x, x1])
  x = layers.LeakyReLU()(x)
  x = layers.BatchNormalization()(x)

  x = layers.Conv2D(filters=32, kernel_size=(2, 5), strides=1, padding="same", use_bias=False, kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = layers.LeakyReLU()(x)
  x = layers.BatchNormalization()(x)

  x = layers.SpatialDropout2D(0.2)(x)
  x = layers.Conv2D(filters=1, kernel_size=[2, FRAME_SIZE - FRAME_OVERLAP + 1], strides=1, padding='same')(x)

  x = layers.Reshape((2, FRAME_SIZE - FRAME_OVERLAP + 2))(x)

  model = Model(inp1, x)
  
  model.compile(optimizer=optimizers.Adam(lr), loss="mse", metrics=[keras.metrics.RootMeanSquaredError('rmse')])
  # model.compile(optimizer=optimizers.Adam(lr), loss="mse", metrics=[SNR])

  return model

if __name__ == "__main__":
  model = create_model(0, 5e-3)
  model.summary()
