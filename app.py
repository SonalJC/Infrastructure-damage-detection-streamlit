import os
import zipfile
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------- PAGE ----------------
st.set_page_config(page_title="Infrastructure Damage Detection", layout="centered")
st.title("🏗️ Infrastructure Damage Detection")
st.write("Upload an image or use your camera to detect structural damage.")

# ---------------- LOAD MODEL ----------------
MODEL_DIR = "damage_model_tf"
MODEL_ZIP = "damage_model_tf.zip"

if not os.path.exists(MODEL_DIR):
    with zipfile.ZipFile(MODEL_ZIP, "r") as z:
        z.extractall(".")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_DIR, compile=False)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ---------------- INPUT ----------------
mode = st.radio("Select Input Method:", ["Upload Image", "Use Webcam"])

image_file = (
    st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if mode == "Upload Image"
    else st.camera_input("Capture image")
)

# ---------------- PREDICTION ----------------
if image_file is not None:
    img = Image.open(image_file).convert("RGB")
    st.image(img, caption="Input Image", use_container_width=True)

    img = img.resize((224, 224))
    img_array = np.array(img).astype("float32")
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Run Detection"):
        pred = model(img_array, training=False).numpy()[0][0]

        st.divider()
        if pred < 0.5:
            st.error(f"⚠️ DAMAGE DETECTED ({(1 - pred) * 100:.2f}%)")
        else:
            st.success(f"✅ NO DAMAGE DETECTED ({pred * 100:.2f}%)")
