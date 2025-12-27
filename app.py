import os
import zipfile
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------- PAGE CONFIG & STYLING ----------------
st.set_page_config(page_title="Infrastructure Damage Detection", layout="centered")

# Custom CSS for Centering
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #808080;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# Centered Title and Subtitle
st.markdown('<div class="main-title">🏗️ AI-Based Structural Damage Detection Using Deep Learning </div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload an image or use your camera to detect structural damage.</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
MODEL_DIR = "damage_model_tf"
MODEL_ZIP = "damage_model_tf.zip"

if not os.path.exists(MODEL_DIR):
    if os.path.exists(MODEL_ZIP):
        with zipfile.ZipFile(MODEL_ZIP, "r") as z:
            z.extractall(".")
    else:
        st.error("❌ Model zip file not found!")
        st.stop()

@st.cache_resource
def load_model():
    model = tf.saved_model.load(MODEL_DIR)
    return model.signatures["serving_default"]

try:
    infer = load_model()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ---------------- INPUT SECTION ----------------
# Radio button ko bhi center feel dene ke liye columns ka use kar sakte hain
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    mode = st.radio("Select Input Method:", ["Upload Image", "Use Webcam"], horizontal=True)

if mode == "Upload Image":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
else:
    image_file = st.camera_input("Capture image")

# ---------------- PREDICTION ----------------
if image_file is not None:
    img = Image.open(image_file).convert("RGB")
    
    # Image Display
    st.image(img, caption="Input Image", use_container_width=True)

    # Preprocessing
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction Button Centering
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        run_btn = st.button("🔍 Run Detection", use_container_width=True)

    if run_btn:
        output = infer(tf.constant(img_array))
        pred = list(output.values())[0].numpy()[0][0]

        st.divider()
        if pred < 0.5:
            st.error(f"### ⚠️ DAMAGE DETECTED ({(1-pred)*100:.2f}%)")
        else:
            st.success(f"### ✅ NO DAMAGE DETECTED ({pred*100:.2f}%)")
