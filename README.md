# 🌱 AgroLens AI

**AgroLens** is an AI-powered web application designed to assist farmers and agronomists in identifying crop diseases early. By uploading a simple leaf photo, the system analyzes patterns using a Deep Learning model to detect diseases like Corn Common Rust, Potato Blight, and more.

## 🚀 Features
* **Multi-Crop Support:** Detects diseases in Corn, Potato, Tomato, and Rice.
* **Deep Learning:** Powered by a custom-trained Convolutional Neural Network (CNN) using PyTorch.
* **Real-time Analysis:** Instant predictions via a Flask REST API.
* **User-Friendly:** Simple, clean web interface for easy uploads.

## 🛠️ Tech Stack
* **Backend:** Python, Flask, PyTorch, Torchvision
* **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
* **Training:** Google Colab (Tesla T4 GPU)
* **Data Processing:** PIL, Numpy

## 📂 Project Structure
```text
AgroLens/
├── backend/
│   ├── app.py           # The Flask API Server
│   ├── training.ipynb   # The Model Training Notebook
│   └── requirements.txt # Python dependencies
├── frontend/
│   └── index.html       # The User Interface
└── README.md