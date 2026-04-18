import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# Set page configuration with a custom theme look
st.set_page_config(
    page_title="Vegetable AI | Precision Detection",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #00b894;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #55efc4;
        color: white;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_stdio=True)

# App Sidebar
st.sidebar.title("🥬 Settings")
st.sidebar.info("This system uses YOLOv11 for real-time vegetable detection.")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)
st.sidebar.markdown("---")
st.sidebar.markdown("### **Author**")
st.sidebar.write("**Habibur Rahman Sajal**")
st.sidebar.write("Computer Science Researcher")

# Main Header
st.title("🥬 Vegetable Detection & Analytics")
st.markdown("---")

# Function to load model
@st.cache_resource
def load_yolo_model():
    # Ensure 'best.pt' is in your GitHub root directory
    model = YOLO('best.pt')
    return model

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Layout columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Upload Image")
    uploaded_file = st.file_uploader("Drop an image of vegetables here...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        raw_image = Image.open(uploaded_file)
        st.image(raw_image, caption="Uploaded Original Image", use_container_width=True)

with col2:
    st.subheader("🔍 Detection Result")
    if uploaded_file:
        with st.spinner("Processing image..."):
            img_array = np.array(raw_image)
            # Run YOLOv11 Inference
            results = model.predict(source=img_array, conf=conf_threshold)
            
            # Plot the results
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="AI Identification Results", use_container_width=True)
            
            # Display Statistics
            st.markdown("### 📋 Statistics")
            labels = []
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                name = model.names[class_id]
                score = float(box.conf[0])
                labels.append(f"{name} ({score:.2%})")
            
            if labels:
                st.success(f"Detected {len(labels)} objects.")
                for label in set(labels):
                    st.write(f"✅ {label}")
            else:
                st.warning("No objects detected at current threshold.")
    else:
        st.info("Please upload an image to start analysis.")

# Footer
st.markdown("---")
st.markdown("<center>Developed for Academic Research Purposes</center>", unsafe_allow_stdio=True)
