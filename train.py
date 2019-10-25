from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import random as rnd
import configparser as cp

# FUNCTIONS

def extract_config():
  config = cp.ConfigParser()
  config.read("config.ini")
  return config

# extracts data from keras and prepares it for training 
def prepare_data():
  # the data, split between train and test sets
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  X = np.concatenate((x_train, x_test), axis=0)
  Y = np.concatenate((y_train, y_test), axis=0)

  size = X.shape[0]
  val_size = int(size * part_val)
  val_args = []

  for x in range(0, val_size):
    i = rnd.randrange(0, size-1)
    while (i in val_args):
      i = rnd.randrange(0, size-1)
    val_args.append(i)

  x_test = X[val_args]
  y_test = Y[val_args]

  X = np.delete(X, val_args, axis=0)
  Y = np.delete(Y, val_args, axis=0)

  if K.image_data_format() == 'channels_first':
      X = X.reshape(X.shape[0], 1, img_rows, img_cols)
      x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
      input_shape = (1, img_rows, img_cols)
  else:
      X = X.reshape(X.shape[0], img_rows, img_cols, 1)
      x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
      input_shape = (img_rows, img_cols, 1)

  X = X.astype('float32')
  X /= 255
  print('X shape:', X.shape)
  print(X.shape[0], 'data samples')
  print('16% of X are test samples')

  # convert class vectors to binary class matrices
  Y = keras.utils.to_categorical(Y, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)
  
  return X, Y, x_test, y_test, input_shape

# prepares model
def prepare_model():
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=input_shape))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25)) #DROPOUT ПОЧИТАТЬ
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))

  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
  
  return model

# saves model to Google Drive
def save_model(model, path_to_save):
  model.save(path_to_save)

# BODY

config = extract_config()

# CONSTANTS

batch_size = int(config.get("constants_for_cnn", "batch_size"))
num_classes = int(config.get("constants_for_cnn", "num_classes"))
epochs = int(config.get("constants_for_cnn", "epochs"))
part_val = float(config.get("constants_for_cnn", "part_val"))
img_rows, img_cols = int(config.get("constants_for_cnn", "img_rows")), int(config.get("constants_for_cnn", "img_cols"))
path_to_model = config.get("pathes", "path_to_model")
  
X, Y, x_test, y_test, input_shape = prepare_data()
model = prepare_model()

model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test)
          )

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

save_model(model, path_to_model)

