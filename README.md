# 🤖 AI Object Detection System (YOLOv8)

A web-based computer vision application that leverages the **YOLOv8 (You Only Look Once)** architecture for real-time object detection. This project demonstrates the practical application of modern Convolutional Neural Networks (CNNs) in a user-friendly interface.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green)

## 📸 Project Overview

This application allows users to upload images which are then processed by a pre-trained neural network. The system visualizes the results by drawing bounding boxes around detected objects, identifying their classes, and displaying confidence scores.

### Key Features:
* **Object Detection:** Capable of recognizing 80+ common object classes (COCO dataset).
* **Adjustable Confidence Threshold:** Real-time slider to filter predictions based on probability scores.
* **Statistical Analysis:** Automatically counts and categorizes detected objects.
* **Tensor Data View:** Access to raw bounding box coordinates and tensors for debugging and analysis.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Interface:** [Streamlit](https://streamlit.io/) (Rapid Web UI Development)
* **Model:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (Nano model for speed)
* **Image Processing:** PIL (Pillow) & NumPy

---

## 🚀 How to Run Locally

Follow these steps to set up the project on your machine:

### 1. Clone the repository
```bash
git clone <YOUR_REPOSITORY_LINK_HERE>
cd My_Object_Detector

```
### 2. Set up a Virtual Environment (Recommended)
```
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies
```
pip install -r requirements.txt

```

### 4. Run the Application
```
streamlit run app.py

```

After running the command, the application will automatically open in your browser at: http://localhost:8501.

📊 Usage Example
Upload: Select an image (JPG/PNG).

Settings: Adjust the Confidence Threshold (e.g., set between 0.25 and 0.80, otherwise you will get False Positives or False Negatives).

Result:
<img width="1920" height="869" alt="image" src="https://github.com/user-attachments/assets/b4b9b105-4d74-4ee6-96d9-085155d44626" />
<img width="1920" height="869" alt="image" src="https://github.com/user-attachments/assets/3500faf2-1da0-4159-a319-412a8d7c7f22" />

Visual output with bounding boxes.

JSON Statistics: {'person': 3, 'dog': 1}.

Inference time: ~0.2 seconds (CPU).

📝 About
This project was developed as part of a coursework/portfolio assignment on the topic: "Machine Learning for Image Recognition: Building a Custom Object Detection System".

License: MIT
