import os
import zipfile
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------- PAGE ----------------
st.set_page_config(page_title="Infrastructure Damage Detection", layout="centered")

st.markdown(
    "<h1 style='text-align:center;'>🏗️ Infrastructure Damage Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Upload an image or use your camera to detect structural damage</p>",
    unsafe_allow_html=True
)

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
    img_array = np.expand_dims((np.array(img) / 127.5 - 1.0), axis=0)

    if st.button("🔍 Run Detection"):
        pred = model.predict(img_array)[0][0]   # ✅ ONLY THIS

        st.divider()

        if pred < 0.33:
            st.error(f"🔴 Severe Damage ({(1-pred)*100:.2f}%)")
        elif pred < 0.66:
            st.warning(f"🟠 Moderate Damage ({(1-pred)*100:.2f}%)")
        else:
            st.success(f"🟢 No / Low Damage ({pred*100:.2f}%)")
