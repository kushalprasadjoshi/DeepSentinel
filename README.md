# DeepSentinel - AI Powered Security Surveillance System 🚨

![DeepSentinel Logo](docs/logo.png) _Logo coming soon_

---

## 🚀 Project Overview

**DeepSentinel** is an end-to-end AI-powered security surveillance solution designed to detect and analyze suspicious activities in educational institutions. Leveraging deep learning and real-time monitoring, DeepSentinel combines visual and structured data for robust threat detection and actionable insights.

---

## 🏆 Why DeepSentinel?

- **Hybrid AI Model:** Fuses image/video analysis with structured incident data for superior accuracy.
- **Real-Time Detection:** Instantly flags suspicious activities from live camera feeds or recorded footage.
- **Intuitive GUI:** User-friendly Tkinter interface for seamless monitoring and alert management.
- **Insightful Analytics:** Visualizes trends, hotspots, and attack types for proactive security planning.
- **Plug-and-Play:** Easy setup—just add your data and start monitoring!

---

## 🛠️ How It Works

### 1. Data Preprocessing
- **Video Processing:** `extract_frames()` converts video clips into images at your chosen FPS.
- **Structured Data:** `ColumnTransformer` handles both numerical and categorical features.
- **Image Processing:** Images are resized and normalized for optimal CNN performance.

### 2. Model Architecture
- **CNN Branch:** Extracts features from images using convolutional layers.
- **Structured Data Branch:** Processes CSV features with dense layers.
- **Fusion Layer:** Merges both branches for holistic incident classification.

### 3. Training Pipeline
- **Efficient Training:** 15 epochs with early stopping for best results.
- **Persistence:** Saves model (`.h5`) and preprocessor (`joblib`) for easy reuse.
- **Validation:** Monitors accuracy to prevent overfitting.

### 4. Visualization & Analytics
- **Monthly Trends:** Track incident frequency over time.
- **Geographical Insights:** Identify top countries by attack count.
- **Attack Types:** Analyze distribution of different attack methods.

### 5. Tkinter GUI Features
- **Live Camera Monitoring:** Real-time detection and alerts.
- **Video File Support:** Analyze previously recorded footage.
- **Alert System:** Visual warnings for detected threats.
- **Status Dashboard:** Live monitoring and system health display.

---

## 📝 Quick Start Guide

### 1. Data Preparation
- Place the **Education Under Attack** dataset in the `dataset/` directory:
    - `education_attacks.csv` (structured data)
    - `images/` (incident images)
    - `videos/` (video clips)

### 2. Model Training
- Run the provided Jupyter notebooks for data processing and model training.
- Visualize dataset characteristics and save your trained model.

### 3. GUI Deployment
- Launch the Tkinter application.
- Select live camera or video file mode.
- Monitor real-time detections and receive instant alerts.

---

## 🌟 Key Features

- Hybrid deep learning model (images + structured data)
- Real-time monitoring with visual alerts
- Historical data analysis and visualization
- Supports both live feeds and recorded videos
- Customizable detection thresholds

---

## 💡 Hackathon Highlights

- **Innovation:** Unique hybrid approach for multi-modal threat detection.
- **Impact:** Empowers educational institutions with AI-driven security.
- **Scalability:** Easily extendable to other domains (corporate, public spaces, etc.).
- **Open Source:** Ready for collaboration and further development.

---

> **DeepSentinel** delivers a powerful, user-friendly, and scalable AI surveillance solution—perfect for hackathons and real-world deployment!
