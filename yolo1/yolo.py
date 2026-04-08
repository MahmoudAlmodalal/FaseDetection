import cv2
import torch
from pathlib import Path

# Load YOLOv5 model (you need to specify the model weights)
weights_path = 'path/to/yolov5s.pt'  # Replace with the path to your YOLOv5 model weights
model = torch.hub.load('ultralytics/yolov5:master', 'custom', path=weights_path)

# Open the video stream or video file
cap = cv2.VideoCapture("C:/Users/p8036/Desktop/face detection/face_detection/face-demographics-walking.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform face detection using YOLOv5
    results = model(frame)  # Detect objects, including faces

    # Filter results to keep only face detections (you may need to adjust labels)
    faces = results.pred[results.pred[:, 5] == 0]

    for face in faces:
        x, y, w, h, confidence = face[:5].cpu().numpy()

        # Draw bounding box around the face
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
