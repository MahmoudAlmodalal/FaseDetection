import cv2
from sortt import Sort
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture('/home/mahmoud/Desktop/face detection/face_detection/video/v7.mp4')  # Replace 'your_video.mp4' with the video file path

# Create the SORT tracker
tracker = Sort()

# Read the first frame
ret, frame = cap.read()

# Select the region of interest (ROI) to track
bbox = cv2.selectROI(cv2.resize(frame, (1000, 650), interpolation=cv2.INTER_AREA), False)
while True:
    # Read a new frame
    ret, frame = cap.read()

    # Break the loop when we reach the end of the video
    if not ret:
        break

    # Update the tracker with the selected ROI as the initial detection
    trackers = tracker.update(np.array([bbox]))
    print(trackers)
    # Draw bounding boxes for tracked objects
    for d in trackers:
        d = [int(x) for x in d]
        cv2.rectangle(frame, (100, 200), (200, 400), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Tracking', cv2.resize(frame, (1000, 650), interpolation=cv2.INTER_AREA))

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
