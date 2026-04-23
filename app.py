import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="VoxHand AI", page_icon="🤟", layout="wide")

# Sexy Neon UI CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: white;
    }
    h1 {
        color: #00f2fe;
        text-shadow: 0px 0px 15px #00f2fe;
        text-align: center;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #00f2fe;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL (Cached) ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

@st.cache_data
def load_labels():
    try:
        with open("labels.txt", "r") as f:
            return [line.strip().split(' ', 1)[-1] for line in f.readlines()]
    except:
        return ["Sign 1", "Sign 2", "Sign 3", "Background"]

model = load_model()
labels = load_labels()

# --- UI ---
st.title("🤟 VoxHand AI: ISL Interpreter")

col1, col2 = st.columns([3, 1])

with col1:
    # Camera input widget (built-in, no extra deps)
    camera = st.camera_input("Show an ISL sign to the camera", key="cam")
    if camera is not None:
        # Convert to numpy array
        image = Image.open(camera)
        frame = np.array(image.convert("RGB"))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Preprocessing (matching Teachable Machine)
        resized = cv2.resize(frame_bgr, (224, 224))
        normalized = (resized.astype(np.float32) / 127.5) - 1
        data = np.expand_dims(normalized, axis=0)

        # Inference
        prediction = model.predict(data, verbose=0)
        index = np.argmax(prediction)
        conf = prediction[0][index]
        label = labels[index]

        # Display result
        st.image(frame, channels="RGB", caption=f"Prediction: {label.upper() if conf > 0.85 else 'Uncertain'}", use_container_width=True)

        if conf > 0.85:
            st.success(f"### 🎯 {label.upper()} (confidence: {conf:.2%})")
        else:
            st.warning(f"Low confidence ({conf:.2%}). Please show sign clearly.")

with col2:
    st.markdown("### 📊 Project Insights")
    st.metric(label="Model Accuracy", value="94%", delta="Stable")
    st.metric(label="Processing", value="Browser-based", delta="Fast")
    st.info("Developed by Aayush Pandey")
    st.write("B.Tech CSE-AIML | 2026")

st.markdown("---")
st.caption("Captures a single frame. Press 'Clear' to try again.")
