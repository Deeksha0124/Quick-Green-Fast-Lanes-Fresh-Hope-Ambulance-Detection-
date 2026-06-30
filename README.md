# 🚑 QuickGreen: Fast Lanes, Fresh Hope

## Overview

QuickGreen: Fast Lanes, Fresh Hope is a final year Artificial Intelligence and Machine Learning project developed by a team of three students. The project focuses on improving emergency vehicle movement in urban traffic by automatically detecting ambulances using computer vision and giving them immediate traffic signal priority.

The system uses a YOLOv8 object detection model to identify ambulances from live video or recorded traffic footage. Once an ambulance is detected, the traffic signal simulation automatically changes from red to green, allowing the ambulance to pass without unnecessary delay. After the ambulance leaves the junction, the traffic signal returns to its normal state.

The project demonstrates how Artificial Intelligence can be applied to intelligent traffic management systems and how existing traffic infrastructure can be enhanced using computer vision.

---

## Problem Statement

Traffic congestion is one of the major challenges in urban areas. Ambulances often lose valuable time while waiting at traffic signals during emergencies. Traditional traffic signals operate on fixed timers and cannot recognize emergency vehicles approaching an intersection.

The objective of this project is to develop a vision based intelligent traffic management system that detects ambulances in real time and provides them with signal priority to reduce delays and improve emergency response time.

---

## Objectives

- Detect ambulances using a YOLOv8 deep learning model.
- Process live video streams and recorded traffic videos.
- Automatically simulate traffic signal changes when an ambulance is detected.
- Restore the traffic signal to its normal operation after the ambulance has passed.
- Demonstrate the practical use of Artificial Intelligence in smart traffic management.

---

## Features

- Real time ambulance detection
- YOLOv8 based object detection
- Automatic traffic signal simulation
- Flask web interface
- Video upload support
- Live detection visualization
- Bounding box detection with confidence score
- Automatic return to normal traffic flow after ambulance detection
- Simple and scalable system architecture

---

## System Architecture

The project consists of the following modules:

- Video Input Module
- Ambulance Detection Module using YOLOv8
- Detection Processing Module
- Traffic Signal Control Module
- Flask Web Application
- Output Visualization Module

---

## Technology Stack

### Programming Language

- Python

### Deep Learning

- YOLOv8
- Ultralytics

### Computer Vision

- OpenCV

### Web Framework

- Flask

### Libraries

- NumPy
- OpenCV
- Ultralytics
- Flask

### Development Tools

- Visual Studio Code
- Jupyter Notebook

### Operating System

- Windows 10 or Windows 11
- Ubuntu 20.04 or later

---

## Project Structure

```
QuickGreen/
│
├── data/
├── implementation/
├── output_images/
├── runs/
├── static/
├── templates/
├── test_images/
├── uploads/
│
├── app.py
├── video_signal.py
├── test.py
├── metric.py
├── best.pt
├── demo_video.mp4
├── output_video.mp4
└── README.md
```

---

## Hardware Requirements

- Intel Core i5 Processor or above
- Minimum 8 GB RAM
- NVIDIA GPU recommended for faster inference
- At least 5 GB free storage

---

## Software Requirements

- Python 3.8 or above
- Windows 10 or Windows 11
- Ubuntu 20.04 or above
- Visual Studio Code or Jupyter Notebook

---

## Installation

### Clone the repository

```bash
git clone https://github.com/your-username/Quick-Green-Fast-Lanes-Fresh-Hope-Ambulance-Detection.git
```

### Move into the project directory

```bash
cd Quick-Green-Fast-Lanes-Fresh-Hope-Ambulance-Detection
```

### Install the required packages

```bash
pip install -r requirements.txt
```

If a requirements.txt file is not available, install the following packages manually.

```bash
pip install ultralytics
pip install opencv-python
pip install flask
pip install numpy
```

---

## Usage

Start the Flask application.

```bash
python app.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

Upload a traffic video or use the provided demo video.

The system will

- Detect ambulances
- Draw bounding boxes
- Display confidence scores
- Change the traffic signal to green
- Restore the signal after the ambulance leaves the scene

---

## Model

The project uses a custom trained YOLOv8 model.

Model file

```
best.pt
```

The model has been trained specifically for ambulance detection using a curated dataset of ambulance images.

---

## Output

The system produces

- Ambulance detection with bounding boxes
- Confidence score for each detection
- Traffic signal simulation
- Processed output video
- Detection images stored in the output directory

---

## Applications

- Smart city traffic management
- Emergency vehicle prioritization
- Intelligent transportation systems
- AI based traffic monitoring
- Urban traffic management research

---

## Future Scope

- Integration with real traffic signal controllers
- Support for multiple emergency vehicles
- Live CCTV camera integration
- GPS based ambulance tracking
- Cloud based deployment
- Edge AI implementation for faster processing
- Improved detection using larger datasets

---

## Team Members

This project was developed by

- D R Neha
- Deeksha M R
- Harshitha N

Department of Artificial Intelligence and Machine Learning

B.N.M Institute of Technology

Batch 2022 to 2026

---

## Project Guide

Abhilasha P Kumar

Assistant Professor

Department of Artificial Intelligence and Machine Learning

B.N.M Institute of Technology

---

## License

This project was developed for academic and educational purposes as part of the Bachelor of Engineering final year project.
