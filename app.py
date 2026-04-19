import os
import io
import base64
from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Load Model
try:
    model = YOLO('best.pt')
except Exception as e:
    print(f"Error loading model: {e}")

# Professional HTML, CSS, and JS all in one string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vegetable AI Researcher</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Poppins', sans-serif; background: #f4f7f6; margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .container { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); width: 90%; max-width: 500px; text-align: center; }
        h1 { color: #2d3436; margin-bottom: 10px; font-size: 24px; }
        h1 span { color: #00b894; }
        p { color: #636e72; font-size: 14px; margin-bottom: 20px; }
        .upload-area { border: 2px dashed #00b894; padding: 20px; border-radius: 10px; cursor: pointer; margin-bottom: 20px; transition: 0.3s; }
        .upload-area:hover { background: #f0fff4; }
        input[type="file"] { display: none; }
        .btn { background: #00b894; color: white; border: none; padding: 12px 25px; border-radius: 8px; font-weight: 600; cursor: pointer; width: 100%; font-size: 16px; transition: 0.3s; }
        .btn:disabled { background: #b2bec3; cursor: not-allowed; }
        #resultArea { margin-top: 25px; display: none; }
        #outputImage { width: 100%; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        .badge-container { margin-top: 15px; display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
        .badge { background: #e6fffa; color: #00b894; border: 1px solid #00b894; padding: 5px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🥬 Vegetable AI <span>Researcher</span></h1>
        <p>Advanced Identification System by Habibur Rahman Sajal</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <span id="fileName">📁 Click to upload vegetable image</span>
            <input type="file" id="fileInput" accept="image/*" onchange="updateFileName(this)">
        </div>

        <button id="detectBtn" class="btn" onclick="analyzeImage()">Analyze Image</button>

        <div id="resultArea">
            <h3 style="font-size: 18px; color: #2d3436;">Detection Result:</h3>
            <img id="outputImage" src="" alt="Result">
            <div id="badgeContainer" class="badge-container"></div>
        </div>
    </div>

    <script>
        function updateFileName(input) {
            const fileName = input.files[0] ? input.files[0].name : "📁 Click to upload vegetable image";
            document.getElementById('fileName').textContent = fileName;
        }

        async function analyzeImage() {
            const fileInput = document.getElementById('fileInput');
            const btn = document.getElementById('detectBtn');
            const resultArea = document.getElementById('resultArea');
            const badgeContainer = document.getElementById('badgeContainer');

            if (!fileInput.files[0]) {
                alert("Please select an image first!");
                return;
            }

            btn.innerText = "Analyzing... Please wait";
            btn.disabled = true;
            resultArea.style.display = 'none';

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();

                if (data.error) {
                    alert(data.error);
                } else {
                    document.getElementById('outputImage').src = "data:image/jpeg;base64," + data.image;
                    badgeContainer.innerHTML = data.items.map(item => `<span class="badge">✅ ${item}</span>`).join('');
                    resultArea.style.display = 'block';
                    resultArea.scrollIntoView({ behavior: 'smooth' });
                }
            } catch (error) {
                alert("Server timeout or error. Please try a smaller image.");
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
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        # Optimization: imgsz=320 for speed
        results = model.predict(source=img, conf=0.25, imgsz=320)
        
        # Plotting result
        res_plotted = results[0].plot()
        res_img = Image.fromarray(res_plotted.astype('uint8'))
        
        # Encoding result image
        buffered = io.BytesIO()
        res_img.save(buffered, format="JPEG", quality=75)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Detected names
        names = [model.names[int(box.cls[0])] for box in results[0].boxes]

        return jsonify({
            'image': img_str,
            'items': list(set(names)) if names else ["No items detected"]
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
