from ultralytics import YOLO
import cv2
import math
import numpy as np
import motpy as mp

def drawBox(frame, bbox, tracker_id, i = 0):
    r = (0, 0, 255)
    if i == 1:
        r = (0, 255, 0)
    cv2.rectangle(frame, bbox[:2], bbox[2:], (0,255,0), 2) # draw boxes on img
    cv2.putText(frame, f"Tracker ID: {tracker_id}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, r, 2)

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def load_model(m):
    model = YOLO(m)
    return model


# Initialize YOLOv8 model
model = load_model("/home/mahmoud/Desktop/face detection/face_detection/face-detection-yolov8-main/yolov8n-face.pt")

# Open a video capture
cap = cv2.VideoCapture("/home/mahmoud/Desktop/face detection/face_detection/video/v8.mp4")

# Create a motpy Tracker
motpy_tracker = mp.Track(1, [1,2,3,4])

while True:
    timer = cv2.getTickCount()
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break
    # Perform object detection using YOLOv8
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    for result in results:
        boxes = result.boxes.cpu().numpy() # get boxes on cpu in numpy
            # Update the motpy tracker with detected boxes
        motpy_tracker.update(boxes)

    # Perform object detection or obtain bounding boxes as needed
    # For simplicity, let's assume you have a list of detected bounding boxes called 'detected_boxes'



    # Get the updated tracks from the tracker
    tracks = motpy_tracker.active_tracks()

    # Draw bounding boxes and labels on the frame
    for track in tracks:
        bbox = track.box
        tracker_id = track.id

        # Draw bounding box and tracker ID on the frame
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"Tracker ID: {tracker_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # Display the frame with detected and tracked objects using OpenCV
    cv2.imshow("Object Tracking", cv2.resize(frame, (1000, 650), interpolation=cv2.INTER_AREA))

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()
