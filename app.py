import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

# Load ONNX model
MODEL_PATH = "brain_tumor_cnn.onnx"
session = ort.InferenceSession(MODEL_PATH)

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Streamlit UI
st.title("Brain Tumor Classification (ONNX Model)")
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess (match training)
    img = img.resize((128, 128))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

    # Inference
    prediction = session.run([output_name], {input_name: img_array})[0]
    class_index = np.argmax(prediction)

    # Your class names
    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    st.write(f"**Prediction:** {class_names[class_index]}")
