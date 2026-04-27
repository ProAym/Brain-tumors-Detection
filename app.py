import os
# Suppress TensorFlow logs and force CPU-only mode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
# Disable memory-heavy features
tf.config.set_visible_devices([], 'GPU')

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, render_template

app = Flask(__name__)

# --- LOAD TFLITE MODEL ---
# This uses much less RAM than load_model()
interpreter = tf.lite.Interpreter(model_path="models/Brain_tumor.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def model_predict(img_path):
    # Load and resize image
    img = Image.open(img_path).resize((224, 224))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Preprocess (Scale by 1/255.0)
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], x)
    
    # Run inference
    interpreter.invoke()

    # Get results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return classes[np.argmax(output_data)]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "No file uploaded"
        
        f = request.files['file']
        basepath = os.path.dirname(os.path.abspath(__file__))
        upload_path = os.path.join(basepath, 'static', 'uploads')
        
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
            
        file_path = os.path.join(upload_path, f.filename)
        f.save(file_path)

        # Predict using the TFLite function
        result = model_predict(file_path)
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return result
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
