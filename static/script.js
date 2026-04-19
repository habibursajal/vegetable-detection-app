const imageInput = document.getElementById('imageInput');
const detectBtn = document.getElementById('detectBtn');
const resultArea = document.getElementById('resultArea');

detectBtn.onclick = async () => {
    const file = imageInput.files[0];
    if (!file) return alert("Please upload an image first!");

    const formData = new FormData();
    formData.append('file', file);

    detectBtn.innerText = "Analyzing... Please wait";
    detectBtn.disabled = true; // বাটনটি ডিজেবল করে দাও যাতে বারবার ক্লিক না হয়
    resultArea.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
        } else {
            document.getElementById('outputImage').src = `data:image/jpeg;base64,${data.image}`;
            document.getElementById('detectionList').innerHTML = data.items.map(i => `<span class="badge">✅ ${i}</span>`).join(' ');
            resultArea.style.display = 'block';
            // রেজাল্ট দেখার জন্য পেজটি একটু নিচে স্ক্রল করবে
            resultArea.scrollIntoView({ behavior: 'smooth' });
        }
    } catch (error) {
        console.error("Error:", error);
        alert("The server is taking too long or encountered an error. Try a smaller image.");
    } finally {
        detectBtn.innerText = "Analyze Image";
        detectBtn.disabled = false;
    }
};
