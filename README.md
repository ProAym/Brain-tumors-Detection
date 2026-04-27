# NeuroScan AI: Deep Learning for Brain Tumor Classification

NeuroScan AI is a professional clinical analysis terminal that leverages Deep Learning to assist in the detection and classification of brain tumors from MRI scans. The system classifies scans into four distinct categories: **Glioma, Meningioma, Pituitary tumor,** and **Healthy.**



## 🚀 Live Demo
The application is deployed on Render and accessible here:  
**https://brain-tumors-detection-wrjt.onrender.com/** *(Note: As this is hosted on a free tier, the server may take ~30 seconds to "wake up" during the first request.)*

## 🧠 Technical Overview
The core of the system is a Convolutional Neural Network (CNN) designed for spatial feature extraction in medical imagery. The project transitions a research-grade model into a production-ready web application using the Flask micro-framework.

### Key Features:
- **Diagnostic Accuracy:** High-precision classification across multiple tumor pathologies.
- **Asynchronous Inference:** Utilizes AJAX for non-blocking image analysis, ensuring a smooth user experience.
- **Medical UI/UX:** A clean, professional clinical dashboard designed with modern CSS3.
- **Resource Optimized:** Specifically engineered to run within a 512MB RAM constraint.

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow, Keras
- **Backend:** Python, Flask, Gunicorn
- **Frontend:** HTML5, CSS3, JavaScript (jQuery)
- **Deployment:** Render, Git LFS



## ⚡ Engineering Challenges & Solutions
Deploying Deep Learning models to constrained cloud environments presents unique hurdles. This project addresses several critical engineering bottlenecks:

1. **Memory Engineering:** - Switched to `tensorflow-cpu` to eliminate the heavy RAM overhead of GPU drivers.
   - Implemented explicit session clearing (`tf.keras.backend.clear_session()`) to prevent memory leaks during inference.
   
2. **Computational Constraints:**
   - Optimized Gunicorn configuration with a 120-second timeout to accommodate CPU-based matrix multiplication on limited hardware.
   
3. **Pipeline Synchronization:**
   - Corrected a critical $x/255$ scaling mismatch between the training distribution and the production API to ensure diagnostic consistency.

## 📁 Project Structure
```text
├── app.py              # Flask backend and inference logic
├── models/             # Trained deep learning models (.keras)
├── static/
│   ├── css/            # Professional UI styling
│   └── uploads/        # Temporary storage for diagnostic analysis
├── templates/
│   └── index.html      # Main clinical terminal interface
├── requirements.txt    # Production-ready dependencies
└── README.md
```
## 💻 Installation & Usage

### 1. Prerequisites
Ensure you have Python 3.12 installed on your local machine.

### 2. Local Installation
```bash
# Clone the repository
git clone [https://github.com/your-username/brain-tumors-detection.git](https://github.com/your-username/brain-tumors-detection.git)

# Navigate to the project directory
cd brain-tumors-detection

# Install required dependencies
pip install -r requirements.txt
```
## 3. Usage
To run the diagnostic terminal locally:
# Start the Flask server
python app.py

Access the terminal in your browser at http://127.0.0.1:10000.

Upload a brain MRI scan in JPG or PNG format.

Click "Run Diagnostic" to initiate the AI-generated classification.

View the result in the clinical result box.
##
📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
##
Developed by Aymane Computer Engineering Graduate
