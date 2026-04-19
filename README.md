# 🥬 Vegetable AI Researcher: Multi-Class Detection System

![YOLOv11](https://img.shields.io/badge/Model-YOLOv11n-green)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)

This repository contains an advanced **Multi-Vegetable Identification System** developed as part of undergraduate research in Computer Science & Engineering. The system is optimized to identify 19 local varieties from complex environments.

## 🚀 Live Demo
The application is live and hosted on Hugging Face Spaces:
[**Access Live Web App**](https://huggingface.co/spaces/habibursajal/Vegetable-Detection-System)

## 📌 Research Overview
- **Core Model:** YOLOv11 (Nano)
- **Dataset:** 
- **Classification:** Local vegetables.
- **Inference:** Optimized for CPU-based real-time detection using Gradio and OpenCV.

## 📂 Key Files
- `app.py`: Gradio Blocks interface with RGB color-correction and analytics.
- `best.pt`: Trained weights for Vegetable dataset (5.5 MB).
- `requirements.txt`: Minimal dependencies for stable deployment.

## 🛠️ Features
- **Multi-Object Detection:** Capable of detecting and counting several vegetables in a single frame.
- **Analytics:** Provides a summarized count of each detected variety.
- **Professional UI:** Clean and academic layout designed for research presentation.

## 📊 Evaluation Metrics
The model was evaluated based on:
- **mAP@.5:** High precision in overlapping objects.
- **Confusion Matrix:** Minimal misclassification between similar-looking botanical varieties.
- **Real-world Robustness:** Tested against local market lighting conditions.

## 👤 Author
**Habibur Rahman Sajal** *Computer Science & Engineering Researcher* *Dhaka, Bangladesh*

---
*Disclaimer: This project is intended for research purposes only.*
