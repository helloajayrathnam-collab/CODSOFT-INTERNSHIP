# -----------------------------
# Auto-install required packages if not present
# -----------------------------
import subprocess
import sys

required_packages = ['opencv-python', 'face_recognition']
for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# -----------------------------
# Face Detection & Recognition Starts Here
# -----------------------------
import cv2
import face_recognition
import numpy as np
import os

# Load known face image
known_image_path = 'known.jpg'  # Replace with actual known face image
if not os.path.exists(known_image_path):
    print("Please add a known image as 'known.jpg'.")
    exit()

known_image = face_recognition.load_image_file(known_image_path)
known_encoding = face_recognition.face_encodings(known_image)[0]
known_face_names = ["Known Person"]
known_face_encodings = [known_encoding]

# Choose mode: 'image' or 'webcam'
mode = 'webcam'  # Change to 'image' to run on a still image
target_image_path = 'target.jpg'  # Used in image mode

def detect_and_recognize_faces(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    return frame

# Run on image
if mode == 'image':
    if not os.path.exists(target_image_path):
        print("Please add a target image named 'target.jpg'")
        exit()
    image = cv2.imread(target_image_path)
    output = detect_and_recognize_faces(image)
    cv2.imshow("Face Detection & Recognition", output)
    cv2.imwrite("output.jpg", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run on webcam
elif mode == 'webcam':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access webcam")
        exit()

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = detect_and_recognize_faces(frame)
        cv2.imshow("Live Face Detection & Recognition", processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
