from tensorflow.keras import callbacks, utils

from helpers import *
from model import create_model

if __name__ == "__main__":
  train_data_generator = DataGenerator(TRAIN_DATASET_PATH, BATCH_SIZE)
  test_data_generator = DataGenerator(TEST_DATASET_PATH, BATCH_SIZE)

  print(len(train_data_generator))
  print(len(test_data_generator))
  print(train_data_generator[0])

  model = create_model(0, lr=5e-3)
  model.summary()
  utils.plot_model(model, show_shapes=True, expand_nested=True)

  number_of_batches = len(train_data_generator) // STEPES_PER_EPOCH

  backup = BackupCallback("backup.json", best_weights_path="best_weights.h5", monitor_value="val_loss", monitor_value_mode="min")
  lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', mode="min",patience=5,factor=0.5,verbose=1)
  lr_scheduler.best = backup.get_value()
  early_stop = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)

  try:
    if os.path.exists("best_weights.h5") and input("Load best? ") == "y":
      model.load_weights("best_weights.h5")
    elif os.path.exists("checkpoint.h5"):
      print("Loading checkpoint")
      model.load_weights("checkpoint.h5")
    else:
      if PRETRAIN_EPOCHS is not None:
        print("Starting pretrain")
        model.fit(train_data_generator, steps_per_epoch=STEPES_PER_EPOCH, epochs=PRETRAIN_EPOCHS)
        model.save_weights("checkpoint.h5")

    print("Starting training")
    model.fit(train_data_generator, steps_per_epoch=STEPES_PER_EPOCH, epochs=EPOCHS * number_of_batches, initial_epoch=backup.get_start_epoch(),
              callbacks=[lr_scheduler, callbacks.TensorBoard("logs", profile_batch=0), backup, early_stop],
              validation_data=test_data_generator, validation_steps=(((STEPES_PER_EPOCH // 3) if (STEPES_PER_EPOCH // 3) < len(test_data_generator) else None) if STEPES_PER_EPOCH is not None else None))
  except KeyboardInterrupt:
    model.save_weights("checkpoint.h5")
  except Exception as e:
    print(f"Error while training:\n{e}")
    exit(-1)

  model.save("model.h5")