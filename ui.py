import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from imutils.video import VideoStream
from datetime import datetime
import csv
import os
import joblib
import imutils

class FaceRecognitionApp:
    def __init__(self, root, csv_file_path, recognizer, le, detector, embedder):
        self.root = root
        self.csv_file_path = csv_file_path
        self.recognizer = recognizer
        self.le = le
        self.detector = detector
        self.embedder = embedder

        self.vs = None
        self.panel = None
        self.streaming = False

        self.initialize_ui()

    def initialize_ui(self):
        self.today_date = datetime.now().strftime("%y-%m-%d")
        self.root.title("Face Recognition Attendance System")

        self.start_button = ttk.Button(self.root, text="Start Recognition", command=self.start_recognition)
        self.start_button.pack(pady=10)

        self.stop_button = ttk.Button(self.root, text="Stop Recognition", command=self.stop_recognition)
        self.stop_button.pack(pady=10)

        self.quit_button = ttk.Button(self.root, text="Quit", command=self.quit_app)
        self.quit_button.pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

    def start_recognition(self):
        if not self.streaming:
            self.vs = VideoStream(src=0).start()
            self.streaming = True
            self.start_recognition_loop()

    def stop_recognition(self):
        if self.streaming:
            self.vs.stop()
            self.streaming = False

    def quit_app(self):
        self.stop_recognition()
        self.root.destroy()

    def start_recognition_loop(self):
        frame = self.vs.read()
        frame = imutils.resize(frame, width=600)

        name, confidence = self.recognize_face(frame)

        text = f"{name}: {confidence:.2f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = imutils.resize(frame, width=800)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)

        if self.panel is None:
            self.panel = tk.Label(image=frame)
            self.panel.image = frame
            self.panel.pack(side="left", padx=10, pady=10)
        else:
            self.panel.configure(image=frame)
            self.panel.image = frame

        self.root.after(10, self.start_recognition_loop)
    def is_attendance_marked(self, name):

        # Check if the student's name is already marked for today
        with open(self.csv_file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['Date'] == self.today_date and row['Student'] == name:
                    return True
        return False
    def recognize_face(self, frame):
        # Resize the frame to have a width of 600 pixels (while maintaining the aspect ratio)
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # Construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # Apply OpenCV's deep learning-based face detector to localize faces in the input image
        self.detector.setInput(imageBlob)
        detections = self.detector.forward()

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > 0.9:
                # Compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # Ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # Construct a blob for the face ROI, then pass the blob through our face embedding model
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.embedder.setInput(faceBlob)
                vec = self.embedder.forward()

                # Perform classification to recognize the face
                preds = self.recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = self.le.classes_[j]

                # Check if the recognition meets the confidence threshold
                if proba >= 0.9:
                    # Check if the attendance is already marked for today
                    if not self.is_attendance_marked(name):
                        # Mark attendance by adding the student's name to the CSV file
                        with open(self.csv_file_path, 'a', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=['Date', 'Student'])
                            writer.writerow({'Date': datetime.now().strftime("%y-%m-%d"), 'Student': name})
                    return name, proba * 100

        # If no face is recognized or confidence is below the threshold
        return "Unknown", 0.0



def main():
    root = tk.Tk()

    # Set your CSV file path for attendance
    today_date = datetime.now().strftime("%y-%m-%d")
    csv_file_path = f"attendance/{today_date}.csv"

    # Set the field names for the CSV file
    fieldnames = ['Date', 'Student']

    # Create a folder named "attendance" if it doesn't exist
    attendance_folder = "attendance"
    os.makedirs(attendance_folder, exist_ok=True)

    # create a CSV file for attendance with headers if it doesn't exist
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # load serialized face detector
    protoPath = "face_detection_model/deploy.prototxt"
    modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load serialized face embedding model
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

    # load the actual face recognition model along with the label encoder
    recognizer = joblib.load("output/recognizer.joblib")  # Use joblib for loading
    le = joblib.load("output/le.joblib")  # Use joblib for loading

    app = FaceRecognitionApp(root, csv_file_path, recognizer, le, detector, embedder)
    root.mainloop()

if __name__ == "__main__":
    main()
