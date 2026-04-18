from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
from PIL import Image
import io
import base64
import numpy as np

app = Flask(__name__)

# Load the model once when the server starts
model = YOLO('best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Read the uploaded image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # YOLO Inference
        results = model.predict(source=img, conf=0.25)
        
        # Plot results on the image
        res_plotted = results[0].plot()
        
        # Convert the plotted image (numpy array) back to a PIL image
        res_img = Image.fromarray(res_plotted.astype('uint8'))
        
        # Save the result image to a buffer as base64
        buffered = io.BytesIO()
        res_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Extract detected class names and confidence scores
        detected_details = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            name = model.names[class_id]
            conf = float(box.conf[0])
            detected_details.append(f"{name} ({conf:.2%})")

        return jsonify({
            'image': img_str,
            'items': list(set(detected_details)) if detected_details else ["No vegetables detected"]
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Use port from environment variable for deployment compatibility
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
