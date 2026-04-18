# 🥬 Vegetable Detection System using YOLOv11

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![YOLO](https://img.shields.io/badge/YOLO-v11n-green)

This repository contains a real-time **Vegetable Detection System** built with **YOLOv11**. This project is part of an undergraduate research focusing on automated vegetable identification in the context of Bangladesh's agricultural variety.

## 🚀 Live Demo
Access the web app here: [**Live Deployment Link**](YOUR_STREAMLIT_LINK_HERE)

## 📌 Project Overview
- **Architecture:** YOLOv11 (Nano)
- **Deployment:** Streamlit Cloud
- **Dataset:** Custom botanical dataset focused on 28+ classes.
- **Goal:** Precision classification for research and agricultural automation.

## 📂 Repository Structure
- `app.py`: The web application script.
- `best.pt`: Trained model weights (YOLOv11).
- `requirements.txt`: Python dependencies.

## 🛠️ Local Setup
1. Clone: `git clone https://github.com/your-username/your-repo.git`
2. Install: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

## 📊 Evaluation
The model is analyzed using **mAP (mean Average Precision)**, **Confusion Matrix**, and **F1-Score curves** to ensure reliability in identifying similar-looking vegetables (e.g., Snake Gourd vs. Papaya).

## 👤 Author
**Habibur Rahman Sajal** Computer Science Researcher, Dhaka, Bangladesh.
