import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# MUST be from mobilenet_v2 to match your notebook's test_datagen
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 

app = Flask(__name__)

# Ensure the model file name is correct
MODEL_PATH = 'models/Brain_tumor.keras'
model = load_model(MODEL_PATH)

# EXACT index order from your Colab print statements:
# 0: Glioma, 1: healthy, 2: meningioma, 3: pituitary
classes = ['The mass is A tumor and it\'s type is: Glioma', 
           'the mass is healthy', 
           'The mass is A tumor and it\'s type is: meningioma', 
           'The mass is A tumor and it\'s type is: pituitary']

def model_predict(img_path, model):
    # 1. Load with exact target size (224, 224)
    img = image.load_img(img_path, target_size=(224, 224))
    
    # 2. Convert to array
    x = image.img_to_array(img)
    
    # 3. Add batch dimension
    x = np.expand_dims(x, axis=0)
    
    # 4. Apply MobileNetV2 preprocessing
    # This is the most critical step to prevent the "always index 1" error
    x = preprocess_input(x)
    
    # 5. Get Prediction
    preds = model.predict(x)
    pred_class = np.argmax(preds)
    
    # Debug info for your VS Code terminal
    print(f"DEBUG - Raw Predictions: {preds}")
    print(f"DEBUG - Predicted Index: {pred_class}")
    
    return classes[pred_class]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded"
    
    f = request.files['file']
    
    # Ensure the upload directory exists locally
    basepath = os.path.dirname(__file__)
    upload_path = os.path.join(basepath, 'static', 'uploads')
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)
        
    file_path = os.path.join(upload_path, f.filename)
    f.save(file_path)

    # Run prediction
    result = model_predict(file_path, model)
    return result

if __name__ == '__main__':
    app.run(debug=True)