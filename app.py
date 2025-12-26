import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Infrastructure Damage Detection",
    layout="centered"
)

st.title("🏗️ Infrastructure Damage Detection")
st.write("Upload an image or use your camera to detect structural damage.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    # Use the correct .h5 file path
    model_path = "finetuned_damage_model.h5"
    
    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Make sure 'finetuned_damage_model.h5' is in the repo root.")
        st.stop()
    
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ---------------- INPUT MODE ----------------
input_mode = st.radio(
    "Select Input Method:",
    ("Upload Image", "Use Webcam")
)

image_file = None

if input_mode == "Upload Image":
    image_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )
else:
    image_file = st.camera_input("Capture image")

# ---------------- PREDICTION ----------------
if image_file is not None:
    img = Image.open(image_file).convert("RGB")
    st.image(img, caption="Input Image", use_container_width=True)

    # Preprocessing
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Run Detection"):
        prediction = model.predict(img_array)[0][0]

        st.divider()

        if prediction < 0.5:
            confidence = (1 - prediction) * 100
            st.error(f"⚠️ DAMAGE DETECTED\n\nConfidence: {confidence:.2f}%")
        else:
            confidence = prediction * 100
            st.success(f"✅ NO DAMAGE DETECTED\n\nConfidence: {confidence:.2f}%")
