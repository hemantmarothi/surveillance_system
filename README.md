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
```
## Installation
Clone the repository:

```bash
git clone <repository-url>
cd surveillance-system
```
Install the dependencies listed above.
Download the YOLO model file (e.g., yolov8n.pt) and place it in the project directory.

## Usage
Run the application:

```bash
python app.py
```
Enter a name for the face you want to register and click the "Register Face" button. A webcam feed will appear, and you can capture the face by pressing the 'c' key.

After registering faces, you can start the surveillance mode by clicking the "Start Surveillance" button. The system will detect motion and recognize registered faces in real-time.

Press 'q' to exit the surveillance feed.
