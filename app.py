import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page Configuration
st.set_page_config(page_title="Vegetable Detection", layout="centered")

st.title("🥬 Vegetable Detection System")
st.write("Upload an image to identify vegetables using YOLOv11 model.")

# Load the model
@st.cache_resource
def load_model():
    # Ensure 'best.pt' is in the same directory
    model = YOLO('best.pt') 
    return model

model = load_model()

# Image Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("🔍 Detecting...")
    
    # Convert PIL image to numpy array for YOLO
    img_array = np.array(image)
    
    # Run prediction
    results = model.predict(source=img_array, conf=0.25)
    
    # Plot results on the image
    res_plotted = results[0].plot()
    
    # Display the result image with bounding boxes
    st.image(res_plotted, caption='Detected Objects', use_container_width=True)
    
    # Show the list of detected labels
    st.subheader("📋 Detection Results:")
    detected_classes = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        conf_score = float(box.conf[0])
        detected_classes.append(f"**{class_name}** (Confidence: {conf_score:.2f})")
    
    if detected_classes:
        # Show unique detected items
        for item in set(detected_classes):
            st.write(f"- {item}")
    else:
        st.write("No vegetables detected. Try with a different image.")
