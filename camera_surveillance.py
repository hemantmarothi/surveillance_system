# ----------------------------- #
#       Import Dependencies      #
# ----------------------------- #
import cv2
import face_recognition
import tkinter as tk
from tkinter import messagebox, ttk
from ultralytics import YOLO
import time
import torch


# ----------------------------- #
#       Global Variables         #
# ----------------------------- #
# Known face encodings and names list
known_face_encodings = []
known_face_names = []

# ----------------------------- #
#       Load YOLO Model          #
# ----------------------------- #
# Function to load the YOLO model for object detection (runs on GPU if available)
def load_yolo_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model = YOLO(model_path).to(device)
    return model


# ----------------------------- #
#       Face Registration        #
# ----------------------------- #
# Function to register a new face from webcam input
def register_face():
    global known_face_encodings, known_face_names

    # Start capturing from the webcam
    video_capture = cv2.VideoCapture(2)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image.")
            break

        # Display the camera feed
        cv2.imshow("Register Face - Press 'c' to capture", frame)

        # Wait for the user to press 'c' to capture the face
        if cv2.waitKey(1) & 0xFF == ord('c'):
            # Detect and encode face
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            if len(face_encodings) == 0:
                messagebox.showwarning("Warning", "No face detected. Please try again.")
                continue

            # Get the name for the new face
            name = name_entry.get().strip()
            if name == "":
                messagebox.showwarning("Warning", "Please enter a name for the face.")
                continue

            # Check if the captured face matches any existing faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
            if True in matches:
                first_match_index = matches.index(True)
                if known_face_names[first_match_index] == "Suspicious":
                    # Update the name of the existing suspicious identity
                    known_face_names[first_match_index] = name
                    messagebox.showinfo("Success", f"Updated face name to '{name}'.")
                    break
                else:
                    messagebox.showwarning("Warning", "This face is already registered with a different identity.")
                    break
            else:
                # Store the face encoding and name
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(name)
                messagebox.showinfo("Success", f"Face for '{name}' registered successfully.")
                video_capture.release()
                cv2.destroyAllWindows()
                break

    # Release the video capture and close the window
    video_capture.release()
    cv2.destroyAllWindows()


# ----------------------------- #
#        Motion Detection        #
# ----------------------------- #
# Function to detect motion between two frames
def detect_motion(first_frame, current_frame):
    gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate the difference between the first and current frame
    diff = cv2.absdiff(gray_first, gray_current)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect significant motion based on the contour area
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Minimum area to consider as motion
            return True
    return False


# ----------------------------- #
#       Drawing Labels           #
# ----------------------------- #
# Function to draw bounding boxes and labels from YOLO object detection results
def draw_labels(frame, results, model):
    for result in results:
        for box in result.boxes:
            # Extract box coordinates, class ID, and confidence
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy[:4]
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            label = f"{model.names[class_id]} {conf:.2f}"
            color = (0, 255, 0)

            # Draw rectangle and text label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


# ----------------------------- #
#         Process Faces          #
# ----------------------------- #
# Function to process detected faces and compare with known faces
def process_faces(frame, face_locations):
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    face_names = []

    # Compare detected face encodings with known encodings
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            known_face_encodings.append(face_encoding)
            known_face_names.append("Suspicious")

        face_names.append(name)

    return face_names


# ----------------------------- #
#      Start Surveillance        #
# ----------------------------- #
# Main function to start the surveillance system with motion detection and YOLO object detection
def start_surveillance():
    video_source = 2  # Webcam as video source
    yolo_model_path = "yolov8n.pt"  # Path to the YOLO model
    model = load_yolo_model(yolo_model_path)

    cap = cv2.VideoCapture(video_source)
    first_frame = None
    motion_timeout = 10  # Seconds after which motion detection resets
    motion_detected_time = None
    surveillance_mode = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            break

        # Capture the first frame for motion detection
        if first_frame is None:
            first_frame = frame
            continue

        # Motion detection before switching to full surveillance
        if not surveillance_mode:
            motion_detected = detect_motion(first_frame, frame)
            if motion_detected:
                print("Motion detected! Starting full surveillance...")
                surveillance_mode = True
                motion_detected_time = time.time()

        # Full surveillance mode: Object detection and face recognition
        if surveillance_mode:
            results = model(frame, conf=0.5)  # Set confidence threshold for YOLO model
            frame = draw_labels(frame, results, model)

            # Detect and process faces
            face_locations = face_recognition.face_locations(frame)
            face_names = process_faces(frame, face_locations)

            # Draw bounding boxes and names for recognized faces
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            # Stop surveillance mode if no motion is detected for a certain time
            if time.time() - motion_detected_time > motion_timeout:
                print("No motion detected. Stopping surveillance...")
                surveillance_mode = False
                first_frame = None

        # Display the current frame
        cv2.imshow("Surveillance Feed", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ----------------------------- #
#   GUI for Face Registration    #
# ----------------------------- #
# Initialize the main application window
app = tk.Tk()
app.title("Face Registration and Surveillance System")

# Style the application window
style = ttk.Style()
style.configure("TFrame", background="#f0f0f0")
style.configure("TLabel", background="#f0f0f0", font=("Arial", 12))
style.configure("TEntry", font=("Arial", 12))
style.configure("TButton", font=("Arial", 12), padding=10)

# Create a frame for the input and buttons
frame = ttk.Frame(app, padding="20")
frame.pack(padx=10, pady=10)

# Entry field for name
tk.Label(frame, text="Enter Name:").pack(pady=(0, 5))
name_entry = ttk.Entry(frame)
name_entry.pack(pady=(0, 10))

# Register button for face registration
register_button = ttk.Button(frame, text="Register Face", command=register_face)
register_button.pack(pady=(0, 5))

# Start surveillance button
start_surveillance_button = ttk.Button(frame, text="Start Surveillance", command=start_surveillance)
start_surveillance_button.pack(pady=(5, 0))

# Start the Tkinter event loop
app.mainloop()
