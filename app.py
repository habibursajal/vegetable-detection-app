import os
import io
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

try:
    model = YOLO('best.pt', task='detect')
    model.to('cpu')
except Exception as e:
    print(f"Model Loading Error: {e}")

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
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Optimization: Use imgsz=320 to reduce RAM usage and increase speed
        results = model.predict(source=img, conf=0.25, imgsz=320, task='detect')
        
        res_plotted = results[0].plot()
        res_img = Image.fromarray(res_plotted.astype('uint8'))
        
        buffered = io.BytesIO()
        res_img.save(buffered, format="JPEG", quality=75)
        img_str = base64.b64encode(buffered.getvalue()).decode()

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
        print(f"Prediction Error: {str(e)}")
        return jsonify({'error': 'Analysis failed. Please try a smaller image.'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
