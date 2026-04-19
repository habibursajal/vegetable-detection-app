import os
import io
import base64
from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Load Model - Since your model is only 5.5MB, it's very efficient
try:
    model = YOLO('best.pt')
except Exception as e:
    print(f"Error loading model: {e}")

# Single-file UI for maximum speed and zero file-path errors
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vegetable AI Researcher</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Poppins', sans-serif; background: #f4f7f6; margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; }
        .container { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); width: 100%; max-width: 500px; text-align: center; }
        h1 { color: #2d3436; margin-bottom: 5px; font-size: 24px; }
        h1 span { color: #00b894; }
        .upload-area { border: 2px dashed #00b894; padding: 25px; border-radius: 10px; cursor: pointer; margin-bottom: 20px; background: #f9fdfc; transition: 0.3s; }
        .upload-area:hover { background: #f0fff4; }
        .btn { background: #00b894; color: white; border: none; padding: 15px; border-radius: 8px; font-weight: 600; cursor: pointer; width: 100%; font-size: 16px; }
        .btn:disabled { background: #b2bec3; cursor: not-allowed; }
        #resultArea { margin-top: 25px; display: none; }
        #outputImage { width: 100%; border-radius: 10px; border: 3px solid #00b894; }
        .badge { display: inline-block; background: #e6fffa; color: #00b894; border: 1px solid #00b894; padding: 6px 15px; border-radius: 20px; font-size: 14px; font-weight: 600; margin: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🥬 Vegetable AI <span>Researcher</span></h1>
        <p style="color: #636e72; font-size: 14px;">By Habibur Rahman Sajal</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <span id="fileName">📁 Upload Vegetable Photo</span>
            <input type="file" id="fileInput" accept="image/*" style="display:none;" onchange="document.getElementById('fileName').innerText = this.files[0].name">
        </div>

        <button id="detectBtn" class="btn" onclick="analyze()">Analyze Image</button>

        <div id="resultArea">
            <h3 style="font-size: 18px; color: #2d3436;">Result:</h3>
            <img id="outputImage" src="">
            <div id="labelContainer" style="margin-top: 15px;"></div>
        </div>
    </div>

    <script>
        async function analyze() {
            const fileInput = document.getElementById('fileInput');
            const btn = document.getElementById('detectBtn');
            const resultArea = document.getElementById('resultArea');

            if (!fileInput.files[0]) return alert("Please select an image!");

            btn.innerText = "Processing... (Wait 10-20s)";
            btn.disabled = true;
            resultArea.style.display = 'none';

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();

                if (data.error) {
                    alert("Error: " + data.error);
                } else {
                    document.getElementById('outputImage').src = "data:image/jpeg;base64," + data.image;
                    document.getElementById('labelContainer').innerHTML = data.items.map(i => `<span class="badge">✅ ${i}</span>`).join('');
                    resultArea.style.display = 'block';
                }
            } catch (e) {
                alert("Server Timeout. Please try a smaller image or a screenshot.");
            } finally {
                btn.innerText = "Analyze Image";
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'})
    
    file = request.files['file']
    try:
        # Step 1: Force Resize image on arrival to 640px to save RAM
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img.thumbnail((640, 640)) 
        
        # Step 2: Prediction with low internal resolution
        results = model.predict(source=img, conf=0.25, imgsz=320)
        
        # Step 3: Draw boxes and compress result
        res_plotted = results[0].plot()
        res_img = Image.fromarray(res_plotted.astype('uint8'))
        
        buffered = io.BytesIO()
        res_img.save(buffered, format="JPEG", quality=70) # Quality 70 keeps file size small
        img_str = base64.b64encode(buffered.getvalue()).decode()

        names = [model.names[int(box.cls[0])] for box in results[0].boxes]

        return jsonify({
            'image': img_str,
            'items': list(set(names)) if names else ["Nothing detected"]
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
