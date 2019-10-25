from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import configparser as cp

# FUNCTIONS

def extract_config():
  config = cp.ConfigParser()
  config.read("config.ini")
  return config

def load_model(path):
  model = keras.models.load_model(path)
  return model

def load_data():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  if K.image_data_format() == 'channels_first':
      x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
  else:
      x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  y_test = keras.utils.to_categorical(y_test, num_classes)
  return x_test, y_test

def print_confusion_matrix(model, x_test, y_test):
  from sklearn import metrics
  y_pred = model.predict(x_test)
  matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
  print(matrix)

# BODY

config = extract_config()

# CONSTANTS

num_classes = int(config.get("constants_for_cnn", "num_classes"))
img_rows, img_cols = int(config.get("constants_for_cnn", "img_rows")), int(config.get("constants_for_cnn", "img_cols"))

model = load_model(config.get("pathes", "path_to_model"))
x_test, y_test = load_data()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print_confusion_matrix(model, x_test, y_test)