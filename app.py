import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, render_template

app = Flask(__name__)

# --- CONFIGURATION ---
# Ensure your model file is named 'Brain_tumor.tflite' inside the models folder
TFLITE_MODEL_PATH = 'Brain_tumor.tflite'
UPLOAD_FOLDER = 'static/uploads'

# --- LOAD TFLITE MODEL ---
# The Interpreter is much lighter on RAM than the full Keras model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details for the math operations
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class mapping based on Keras alphabetical sorting
classes = [
    "The mass is A tumor and it's type is: Glioma", 
    "The mass is A tumor and it's type is: Meningioma", 
    "The mass is healthy (No Tumor)", 
    "The mass is A tumor and it's type is: Pituitary"
]

def model_predict(img_path):
    """
    TFLite Inference Pipeline
    """
    # 1. Load and Preprocess Image
    img = Image.open(img_path).resize((224, 224))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 2. Convert to float32 and scale (0 to 1)
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0) # Add batch dimension

    # 3. Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], x)
    
    # 4. Run the "Tiny" Inference
    interpreter.invoke()

    # 5. Get the result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_class = np.argmax(output_data)
    
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
        if f.filename == '':
            return "No selected file"

        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
            
        basepath = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(basepath, UPLOAD_FOLDER, f.filename)
        f.save(file_path)

        # Predict using TFLite
        result = model_predict(file_path)
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return result

    except Exception as e:
        return f"TFLite Error: {str(e)}"

if __name__ == '__main__':
    # Render dynamic port binding
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
