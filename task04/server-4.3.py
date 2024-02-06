import tensorflow as tf
from flask import Flask, request
import numpy as np
import urllib
import cv2
import pathlib

app = Flask(__name__)

class_names = ['black_dress', 'black_pants', 'black_shirt', 'black_shoes', 'black_shorts', 'blue_dress', 'blue_pants', 'blue_shirt', 'blue_shoes', 'blue_shorts', 'brown_pants', 'brown_shoes', 'brown_shorts', 'green_pants', 'green_shirt', 'green_shoes', 'green_shorts', 'red_dress', 'red_pants', 'red_shoes', 'white_dress', 'white_pants', 'white_shoes', 'white_shorts']
model = tf.keras.models.load_model("model-4.3\model")

def predict_image_label(input_image):
    pred = model.predict(process_image(input_image))
    return class_names[np.argmax(pred)]

def process_image(image_path):
    x = urllib.request.urlopen(image_path)
    x = np.asarray(bytearray(x.read()), dtype="uint8")
    x = cv2.imdecode(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (224, 224))
    x = np.expand_dims(x, axis=0).astype(np.float32)
    return x

@app.route('/', methods=['POST', 'GET'])
def classify():
    if request.method == 'POST':
        data = request.get_json()
        image_path = data['image_path']
        return {'label': predict_image_label(image_path)}
    else:
        return 'Make POST request with image_path'                

if __name__ == '__main__':
    app.run()