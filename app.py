import os
import io
import base64
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
import gc  # Garbage Collector to free up RAM

app = Flask(__name__)

# Load model with half precision if possible, though CPU ignores it, 
# it's good practice.
model = YOLO('best.pt', task='detect')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    try:
        # 1. Open and resize image IMMEDIATELY to save RAM
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img.thumbnail((416, 416)) # Resize to max 416px before sending to YOLO

        # 2. Run prediction with absolute minimum settings
        results = model.predict(
            source=img, 
            conf=0.35,     # Increased confidence to filter noise
            imgsz=320,     # Very small internal resolution for speed
            task='detect',
            verbose=False,
            device='cpu'   # Force CPU
        )
        
        # 3. Handle results
        res_plotted = results[0].plot()
        res_img = Image.fromarray(res_plotted.astype('uint8'))
        
        # Save with high compression to speed up transfer
        buffered = io.BytesIO()
        res_img.save(buffered, format="JPEG", quality=60) 
        img_str = base64.b64encode(buffered.getvalue()).decode()

        detected_items = [model.names[int(box.cls[0])] for box in results[0].boxes]

        # 4. Clear memory manually
        del results
        gc.collect() 

        return jsonify({
            'image': img_str,
            'items': list(set(detected_details)) if (detected_items := detected_items) else ["Nothing detected"]
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Server Busy. Try a smaller image.'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
