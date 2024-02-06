import tensorflow as tf
from flask import Flask, request
import numpy as np
import urllib
import cv2
import pathlib
import requests

app = Flask(__name__)

class_names = ['black', 'blue', 'brown', 'dress', 'green', 'pants', 'red', 'shirt', 'shoes', 'shorts', 'white']
model = tf.keras.models.load_model("model-4.4\model")

def image_preds(input_image):
    preds = model.predict(process_image(input_image))
    return preds[0]

def process_image(image_path):
    x = requests.get(image_path, stream = True).raw
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
        preds = image_preds(image_path)
        top_2 = np.argsort(preds)[:-4:-1]
        return [
            {
                'label': class_names[top_2[0]], 
                'score': '{:.3}'.format(preds[top_2[0]])
            },
            {
                'label': class_names[top_2[1]], 
                'score': '{:.3}'.format(preds[top_2[1]])
            }          
        ]
    else:
        return 'Make POST request with image_path'                               

if __name__ == '__main__':
    app.run()