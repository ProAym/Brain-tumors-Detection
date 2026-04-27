import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load your model (ensure the file is in the models/ folder)
MODEL_PATH = 'models/best_brain_model.keras'
model = load_model(MODEL_PATH)

# Classes identified in your notebook
classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

def model_predict(img_path, model):
    # Match the target_size=(224, 224) from your notebook
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x) # Use the same preprocessing as training
    
    preds = model.predict(x)
    return classes[np.argmax(preds)]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded"
    
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'static/uploads', f.filename)
    f.save(file_path)

    # Get Prediction
    result = model_predict(file_path, model)
    return result

if __name__ == '__main__':
    app.run(debug=True)