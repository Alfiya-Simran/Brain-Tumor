import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("brain_tumor_cnn.keras")  # Ensure this file exists in your working dir

st.title("Brain Tumor Classification")
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

    # Preprocess Image (match training settings)
    img = image.load_img(uploaded_file, target_size=(128, 128))  # match training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # same normalization as training

    # Prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction, axis=1)[0]

    # Replace with your own class names
    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    st.write(f"**Prediction:** {class_names[class_index]}")
