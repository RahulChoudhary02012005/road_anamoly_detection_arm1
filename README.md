# Road Anomaly Detection using YOLOv8 and TensorFlow Lite (ARM Deployment)

## Overview

This project implements a real-time Road Anomaly Detection system using a custom-trained YOLOv8 deep learning model. The system detects road anomalies such as potholes, cracks, and surface damage from video or camera input.

The trained model is exported into multiple optimized formats including PyTorch (.pt), ONNX (.onnx), and TensorFlow Lite (.tflite) for deployment on ARM-based edge devices such as Raspberry Pi 4.

This project enables efficient edge AI inference with optimized performance and low resource usage.

---

## Project Objectives

- Train a custom YOLOv8 model on a road anomaly dataset  
- Export the trained model to ONNX format  
- Convert the model to TensorFlow Lite (FP16 and FP32)  
- Optimize the model for ARM architecture  
- Deploy on Raspberry Pi 4  
- Perform real-time anomaly detection  

---

## Model Information

Model Architecture: YOLOv8  
Framework: Ultralytics YOLOv8  

Available Models:

- models/best.pt (Final trained model)
- models/best.onnx (ONNX export)
- models/best_float16.tflite (FP16 optimized model)
- models/best_float32.tflite (FP32 model)
- models/yolo26n.pt (Custom trained model)

Input Resolution: 640x640  
Output: Bounding boxes, labels, and confidence scores  

Deployment Target: Raspberry Pi 4 (ARM)

---

## Repository Structure

road_anamoly_detection_arm/

.gitignore  
README.md  
requirements.txt  

models/  
best.pt  
best.onnx  
best_float16.tflite  
best_float32.tflite  
yolo26n.pt  

scripts/  
convert_int8.py  
float_16.py  
args.yaml  
metadata.yaml  
2.yaml  

dataset/  
README.md  
sample_images/  

results/  
sample_output.mp4  

docs/  
architecture.png  

---

## Installation

Clone the repository:

git clone https://github.com/RahulChoudhary02012005/road_anamoly_detection_arm.git

Navigate to the project directory:

cd road_anamoly_detection_arm

Install dependencies:

pip install -r requirements.txt

Or install manually:

pip install ultralytics tensorflow opencv-python numpy onnx onnxruntime tflite-runtime

---

## Model Training

Train the YOLOv8 model using:

yolo detect train data=scripts/2.yaml model=yolov8n.pt epochs=100 imgsz=640

The trained model will be saved as:

models/best.pt

---

## Model Export

Export to ONNX format:

yolo export model=models/best.pt format=onnx

Convert to TensorFlow Lite FP16:

python scripts/float_16.py

Convert to TensorFlow Lite INT8:

python scripts/convert_int8.py

---

## Laptop Inference

Run inference using PyTorch model:

yolo predict model=models/best.pt source=video.mp4

---

## Raspberry Pi Inference (ARM)

Install dependencies on Raspberry Pi:

pip install tflite-runtime opencv-python numpy

Run inference script:

python scripts/inference.py

---

## Dataset Information

The model was trained on a custom Road Anomaly Detection dataset in YOLO format.

Dataset structure:

images/  
labels/  

Full dataset download link (Google Drive Folder):

https://drive.google.com/drive/folders/1GjtcUAMgKa5iI80LIDEp_gBC2xdVrZzf?usp=sharing

Download the dataset and place it in the following structure:

dataset/

dataset/images/  
dataset/labels/  

Sample dataset images are available in:

dataset/sample_images/

---

## Optimization Techniques Used

- ONNX export  
- TensorFlow Lite conversion  
- FP16 optimization  
- INT8 quantization  
- ARM edge optimization  

These optimizations reduce model size and improve inference speed on edge devices.

---

## System Workflow

1. Capture video from camera or file  
2. Extract frames using OpenCV  
3. Preprocess frames  
4. Run model inference  
5. Detect road anomalies  
6. Draw bounding boxes and labels  
7. Display or save output  

---

## Hardware Used

- Raspberry Pi 4 (8GB RAM)  
- USB Camera or Pi Camera  
- Laptop for training and conversion  

---

## Software Used

- Python 3.10  
- Ultralytics YOLOv8  
- TensorFlow Lite  
- ONNX Runtime  
- OpenCV  
- NumPy  

---

## Results

Sample output video is available in:

results/sample_output.mp4

---

## Applications

- Smart road monitoring  
- Autonomous vehicle assistance  
- Driver safety systems  
- Infrastructure maintenance  
- Edge AI deployment  

---

## Author

Rahul Choudhary  
Road Anomaly Detection using YOLOv8 on ARM  

GitHub:  
https://github.com/RahulChoudhary02012005

---

## License

This project is licensed under the MIT License.
