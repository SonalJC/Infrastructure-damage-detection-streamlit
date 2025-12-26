import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image



# --- Page Settings ---
st.set_page_config(page_title="Damage Detector", layout="centered")

# --- Load Model ---


@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "damage_model_tf")
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

st.title("🏗️ Infrastructure Damage Detection")
st.write("Upload a photo or use your camera to check for structural damage.")

# --- Choice Menu ---
input_mode = st.radio("Select Input Method:", ("Upload Image", "Use Webcam"))

source_img = None

if input_mode == "Upload Image":
    source_img = st.file_uploader("Pick an image...", type=["jpg", "jpeg", "png"])
else:
    source_img = st.camera_input("Take a photo")

# --- Logic ---
if source_img is not None:
    img = Image.open(source_img)
    st.image(img, caption="Target Image", use_container_width=True)
    
    # Preprocessing
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype('float32')
    
    # Handle PNG (remove Alpha channel if it exists)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1.0  # Normalization for MobileNetV2
    
    if st.button("Run Detection"):
        prediction = model.predict(img_array)[0][0]
        
        st.divider()
        if prediction < 0.5:
            score = (1 - prediction) * 100
            st.error(f"⚠️ DAMAGE DETECTED ({score:.2f}% Confidence)")
        else:
            score = prediction * 100
            st.success(f"✅ NO DAMAGE FOUND ({score:.2f}% Confidence)")