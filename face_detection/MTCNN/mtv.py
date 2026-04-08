import cv2
import numpy as np
from norfair import Detection, Tracker, Video, draw_points
from mtcnn import MTCNN

# Initialize the MTCNN face detector
face_detector = MTCNN()

# Norfair
video = Video(input_path="/home/mahmoud/Desktop/face detection/face_detection/video/v8.mp4")
cap = cv2.VideoCapture("C:/Users/p8036/Desktop/face detection/face_detection/video/v4.mp4")
tracker = Tracker(distance_function="euclidean", distance_threshold=20)

for frame in video:
    timer = cv2.getTickCount()
    ret, f = cap.read()

    # Detect faces using MTCNN
    detections = []
    faces = face_detector.detect_faces(frame)
    for face in faces:
        # Convert the face detections to Norfair Detection objects
        detection = Detection(np.array([face['box'][0] + face['box'][2], face['box'][1] + face['box'][3] / 2]))

        # Convert detections to a list of lists
        detections.append(detection)

    # Update the tracker with the face detections
    tracked_objects = tracker.update(detections=detections)

    # Show the original frame if it's not empty and has valid dimensions
    if not f is None and f.shape[0] > 0 and f.shape[1] > 0:
        cv2.imshow("Original Frame", f)

    # Draw the tracked objects on the frame
    frame_with_tracks = draw_points(frame, tracked_objects)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Tracked Objects", frame_with_tracks)

    # Check for user input to exit
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
