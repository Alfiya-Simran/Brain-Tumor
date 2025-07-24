# 🧠 Brain Tumor Classification using MRI Images

## 📌 Project Type
Classification, Deep Learning, Computer Vision

## 👩‍💻 Contributor
**Alfiya Simran**

---

## 📖 Project Summary
This project aims to classify brain MRI scans into four categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor** using deep learning.  
The goal is to assist in the early detection of brain tumors, enabling medical professionals to make faster and more accurate decisions.

A Convolutional Neural Network (CNN) model was trained on preprocessed MRI images. Data augmentation techniques were applied to improve model generalization. The model is converted to **ONNX format** for lightweight deployment without requiring TensorFlow, making it easier to run in a variety of environments.

The deployed application is built using **Streamlit** and allows users to upload MRI images and get instant predictions.

---

## 🚀 Live Demo
[**Click here to use the app**](https://brain-tumor-bxvxknrepuc57hq7hxrhef.streamlit.app/)

---

## 🗂 Dataset
- Source: https://drive.google.com/drive/folders/1C9ww4JnZ2sh22I-hbt45OR16o4ljGxju?usp=drive_link

---

## 🛠 Tech Stack
- **Python**: Data Processing & Model Training
- **TensorFlow / Keras**: Initial Model Training
- **ONNX & ONNXRuntime**: Lightweight Model Inference
- **Streamlit**: Web Application Deployment

---

## 📈 Model Performance
- Training Accuracy: ~87%
- Test Accuracy: ~86.9%
- Optimized for real-time predictions with minimal latency

---

## 📂 Repository Structure

Brain-Tumor-Classification/

├── app.py # Streamlit app

├── brain_tumor_cnn.onnx # Trained ONNX model

├── requirements.txt # Dependencies

├── README.md # Project documentation

└── notebooks/ # Jupyter notebooks (EDA & training)

---

## ⚙️ How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/Alfiya-Simran/Brain-Tumor
   cd Brain-Tumor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open http://localhost:8501 in your browser.

---

📜 License
This project is open source and available under the MIT License.

---
