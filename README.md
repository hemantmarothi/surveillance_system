# surveillance_system
# Face Registration and Surveillance System

This project implements a Face Registration and Surveillance System using Python, OpenCV, face recognition, and YOLO (You Only Look Once) for object detection. The system allows users to register faces and start a surveillance mode that detects motion and recognizes registered faces in real time.

## Table of Contents

- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Functionality](#functionality)
- [Challenges and Solutions](#challenges-and-solutions)
- [Contributing](#contributing)
- [License](#license)

## Features

- Register faces from a webcam.
- Detect motion and start surveillance mode.
- Recognize registered faces using face encodings.
- Utilize YOLO for object detection.
- User-friendly GUI for interaction.

## Dependencies

This project requires the following Python packages:

- OpenCV (`cv2`)
- face_recognition
- tkinter
- ultralytics (for YOLO)
- torch (for PyTorch)

You can install the required packages using pip:

```bash
pip install opencv-python face_recognition tkinter ultralytics torch
