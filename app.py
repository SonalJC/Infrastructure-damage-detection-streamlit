import os
import zipfile
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Infrastructure Damage Detection",
    layout="centered"
)

# ---------------- CENTERED TITLE ----------------
st.markdown(
    """
    <h1 style='text-align:center;'>🏗️ Infrastructure Damage Detection</h1>
    <p style='text-align:center; font-size:18px; color:gray;'>
        Upload an image or use your camera to detect structural damage
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------- MODEL SETUP ----------------
MODEL_DIR = "damage_model_tf"
MODEL_ZIP = "damage_model_tf.zip"

# unzip model if not exists
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
mode = st.radio(
    "Select Input Method:",
    ["Upload Image", "Use Webcam"],
    horizontal=True
)

image_file = (
    st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if mode == "Upload Image"
    else st.camera_input("Capture image")
)

# ---------------- PREDICTION ----------------
if image_file is not None:
    img = Image.open(image_file).convert("RGB")
    st.image(img, caption="Input Image", use_container_width=True)

    # preprocessing
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Run Detection"):
        pred = model(img_array, training=False).numpy()[0][0]
        confidence = pred * 100

        # severity logic
        if pred < 0.4:
            severity = "🔴 Severe Damage"
            explanation = "Major structural cracks or failures detected. Immediate inspection required."
        elif pred < 0.7:
            severity = "🟠 Moderate Damage"
            explanation = "Visible structural damage detected. Maintenance or repair recommended."
        else:
            severity = "🟢 Low / No Damage"
            explanation = "No significant structural damage detected. Structure appears safe."

        # result card
        st.markdown(
            f"""
            <div style="
                background-color:#111827;
                padding:25px;
                border-radius:15px;
                text-align:center;
                margin-top:30px;
                color:white;
            ">
                <h2>{severity}</h2>
                <h3>Confidence: {confidence:.2f}%</h3>
                <p style="font-size:16px;">{explanation}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
