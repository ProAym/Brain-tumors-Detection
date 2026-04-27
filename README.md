# NeuroScan AI: Deep Learning System for Brain Tumor Classification

NeuroScan AI is a professional clinical analysis terminal designed to assist in the diagnosis of neurological pathologies. Utilizing a Deep Learning backbone, the system provides automated classification of MRI scans into four distinct categories: **Glioma, Meningioma, Pituitary tumor,** and **Healthy.**

---

## 📋 Project Overview
The primary objective of this project is to bridge the gap between academic research and accessible medical diagnostics. By transforming a trained Convolutional Neural Network (CNN) into a production-ready web application, NeuroScan AI demonstrates how Computer Engineering principles can be applied to life-saving medical technology.

### Pathological Scope:
* **Glioma:** Malignant tumors originating in the glial cells of the brain.
* **Meningioma:** Primary central nervous system (CNS) tumors arising from the meninges.
* **Pituitary Tumors:** Abnormal growths that develop in the pituitary gland.
* **Healthy:** MRI scans showing normal neurological structures.

---

## 🧠 Technical Methodology

### Model Architecture
The system utilizes a Sequential Convolutional Neural Network (CNN) architecture optimized for spatial feature extraction. The model architecture includes:
- **Feature Extraction Layers:** Multiple Convolutional layers using 3x3 kernels for edge and density detection.
- **Dimensionality Reduction:** Max Pooling layers to preserve critical diagnostic features while reducing computational load.
- **Classification Head:** Dense layers with Softmax activation to provide a probability distribution across classes.

### Pipeline Synchronization & Normalization
A critical engineering challenge was the synchronization of the preprocessing pipeline. To ensure 100% diagnostic consistency between the training environment and production, we implemented strict Min-Max scaling:
$$f(x) = \frac{x}{255.0} \in [0, 1]$$

---

## ⚡ Cloud Engineering & Optimization
Deploying high-memory models on a constrained environment (Render Free Tier - 512MB RAM) required extensive **Memory Engineering**:

1.  **Library Optimization:** Switched to `tensorflow-cpu` to strip out heavy GPU-specific binaries (CUDA/cuDNN).
2.  **Session Management:** Implemented `tf.keras.backend.clear_session()` after every prediction to free up the RAM buffer.
3.  **Environment Shaping:** Forced CPU-only mode via `os.environ` to prevent unnecessary hardware searches.
4.  **Timeout Configuration:** Adjusted Gunicorn timeouts to **120 seconds** to ensure successful inference on limited CPU hardware.

---

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Backend:** Python, Flask, Gunicorn
* **Frontend:** HTML5, CSS3 (Clinical Theme), JavaScript (jQuery/AJAX)
* **Version Control:** Git LFS (Large File Storage)
* **Hosting:** Render

---

## 📁 Project Structure
```text
├── app.py              # Main Flask application & Inference logic
├── models/             # Trained .keras model (Stored via Git LFS)
├── static/
│   ├── css/            # Medical-grade UI styling
│   └── uploads/        # Temporary diagnostic storage
├── templates/
│   └── index.html      # Clinical terminal dashboard
├── requirements.txt    # Production dependencies
└── README.md           # Technical documentation
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
