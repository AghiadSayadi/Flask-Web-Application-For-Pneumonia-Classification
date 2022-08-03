import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

IMG_SIZE = 200

app = Flask(__name__)

# Loading our model
model = load_model('CNNModel.h5')

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = img/255
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    print(result)
    if result>= 0.5:
        result="Pneumonia"
    else:
        result="Normal"
    return result

# Main Page 
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Result Page
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'UploadedImages', secure_filename(file.filename))
        file.save(file_path)
        result = model_predict(file_path, model)
        return result
    return None

if __name__ == '__main__':
    app.run(port=5000,debug=True)
