from ultralytics import YOLO
import cv2
import numpy as np
from norfair import Detection, Tracker, Video, draw_points
import time


# Norfair
video = Video(input_path="/home/mahmoud/Desktop/face detection/face_detection/video/v4.mp4")
cap = cv2.VideoCapture("/home/mahmoud/Desktop/face detection/face_detection/video/v4.mp4")
model = YOLO("/home/mahmoud/Desktop/face detection/face_detection/face-detection-yolov8-main/yolov8n-face.pt")
tracker = Tracker(distance_function="euclidean", distance_threshold=60, detection_threshold = 1, reid_distance_threshold = 5, reid_hit_counter_max = 0)
s = time.time()
for frame in video:
    timer = cv2.getTickCount()
    _, f = cap.read()
    frame = cv2.resize(frame, (int(frame.shape[1] * 0.25), int(frame.shape[0] * 0.25)))
    detections = []
    faces = model(frame)
    # Draw bounding boxes and labels on the frame
    for face in faces:
        boxes = face.boxes.cpu().numpy() # get boxes on cpu in numpy
        for box in boxes: # iterate boxes
            
            r = box.xyxy[0].astype(int) # get corner points as int
            # Convert the face detections to Norfair Detection objects
            detection = Detection(np.array([r[0] + (r[2] - r[0]) / 2 , r[1] + (r[3] - r[1]) / 2]))

            # Convert detections to a list of lists
            detections.append(detection)

    # Update the tracker with the face detections
    tracked_objects = tracker.update(detections=detections)


    # Draw the tracked objects on the frame
    frame_with_tracks = draw_points(frame, tracked_objects, text_size = 0.75)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Tracked Objects", frame_with_tracks)

    # Check for user input to exit
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break
print(time.time() - s)
cap.release()
cv2.destroyAllWindows()
