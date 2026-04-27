import os
# Suppress TensorFlow logs and force CPU-only mode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
# Disable memory-heavy features
tf.config.set_visible_devices([], 'GPU')

import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = 'models/Brain_tumor.keras'
UPLOAD_FOLDER = 'static/uploads'

# Load the trained model
model = load_model(MODEL_PATH)

# Exact class mapping from your Colab training:
# 0: Glioma, 1: Healthy, 2: Meningioma, 3: Pituitary
classes = [
    "The mass is A tumor and it's type is: Glioma", 
    "The mass is healthy", 
    "The mass is A tumor and it's type is: meningioma", 
    "The mass is A tumor and it's type is: pituitary"
]

def model_predict(img_path, model):
    """
    Processes the image and returns the model's prediction.
    """
    # 1. Load and resize to match model input (224x224)
    img = load_img(img_path, target_size=(224, 224))
    
    # 2. Convert to array and scale (0 to 1) 
    # This scaling was the key to fixing the prediction error!
    x = img_to_array(img) / 255.0
    
    # 3. Add batch dimension
    x = np.expand_dims(x, axis=0)
    
    # 4. Run Inference
    predictions = model.predict(x)
    pred_class = np.argmax(predictions, axis=1)[0]
    
    return classes[pred_class]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return "No file uploaded"
        f = request.files['file']
        
        # Save file
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static', 'uploads')
        if not os.path.exists(upload_path): os.makedirs(upload_path)
        file_path = os.path.join(upload_path, f.filename)
        f.save(file_path)

        # Predict
        result = model_predict(file_path, model)
        
        # Clean up
        if os.path.exists(file_path): os.remove(file_path)
        
        return result
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}") # This will show in Render logs
        return f"Error: {str(e)}"

if __name__ == '__main__':
    # Use Render's port environment variable or default to 10000
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
