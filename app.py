import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- PAGE SETTINGS ---
st.set_page_config(page_title="🏗️ Structural Damage Detector", layout="centered")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("damage_model_tf", compile=False)

model = load_model()

# --- PREDICTION FUNCTION ---
def predict(image):
    # Resize image to 224x224 (MobileNetV2 input size)
    img = image.resize((224, 224))
    
    # Convert to array
    img_array = np.array(img)
    
    # Preprocess like MobileNetV2
    img_array = preprocess_input(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Ensure correct dtype
    img_array = img_array.astype(np.float32)
    
    # Predict
    output = model.predict(img_array)
    return output[0][0]  # return single value

# --- STREAMLIT UI ---
st.title("🏗️ Structural Damage Detection")
st.write("Upload an image of a structure to detect potential damage.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing..."):
        damage_prob = predict(image)
    
    # Display result
    if damage_prob > 0.5:
        st.error(f"⚠️ Damage Detected! Probability: {damage_prob:.2f}")
    else:
        st.success(f"✅ No Significant Damage Detected. Probability: {damage_prob:.2f}")
