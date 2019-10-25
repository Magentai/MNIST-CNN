from __future__ import print_function
import keras
import json
from keras import backend as K
import numpy as np
from PIL import Image
import configparser as cp

# FUNCTIONS

def extract_config():
  config = cp.ConfigParser()
  config.read("config.ini")
  return config

def load_model(path_to_model):
  model = keras.models.load_model(path_to_model)
  return model

def load_and_prepare_data(path_to_test_data):
  data = []
  for i in range(10):
    img = Image.open(path_to_test_data+str(i)+".png")
    temp = list(img.getdata())
    data.append(temp)
  data = np.array(data)
  data = data.reshape(data.shape[0], 28, 28, 1)
  data = data.astype('float32')
  data /= 255
  return data

def predict_data(model):
  pred = model.predict(data)
  values = pred.argmax(axis=0).tolist()
  pred = pred.tolist()
  return values, pred

def create_json(values, pred):
  json_list = []
  for i in range(10):
    x = {}
    x["expected"] = i
    x["predicted"] = values[i]
    x["result"] = pred[i]
    json_list.append(x)
  with open(path_to_json, 'w', encoding='utf-8') as f:
      for i in range(len(json_list)):
        json.dump({"expected": json_list[i]["expected"]}, f, ensure_ascii=False, indent=4)
        json.dump({"predicted": json_list[i]["predicted"]}, f, ensure_ascii=False, indent=4)
        json.dump({"result": json_list[i]["result"]}, f, ensure_ascii=False, indent=4)
        
# BODY

config = extract_config()

# CONSTANTS

path_to_model = config.get("pathes", "path_to_model")
path_to_test_data = config.get("pathes", "path_to_test_data")
path_to_json = config.get("pathes", "path_to_json")

model = load_model(path_to_model)
data = load_and_prepare_data(path_to_test_data)
values, pred = predict_data(model)
create_json(values, pred)