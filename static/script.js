const imageInput = document.getElementById('imageInput');
const detectBtn = document.getElementById('detectBtn');
const resultArea = document.getElementById('resultArea');

detectBtn.onclick = async () => {
    const file = imageInput.files[0];
    if (!file) return alert("Please upload an image first!");

    const formData = new FormData();
    formData.append('file', file);

    detectBtn.innerText = "Analyzing...";
    
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    
    document.getElementById('outputImage').src = `data:image/jpeg;base64,${data.image}`;
    document.getElementById('detectionList').innerHTML = data.items.map(i => `<span>✅ ${i}</span>`).join(' ');
    resultArea.style.display = 'block';
    detectBtn.innerText = "Analyze Image";
};
