import dlib
import cv2

#170 y
def drawBox(frame, bbox, tracker_id, i=0):
    r = (0, 0, 255)
    if i == 1:
        r = (0, 255, 0)
    cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 255, 0), 2)  # draw boxes on img
    cv2.putText(frame, f"Tracker ID: {tracker_id}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, r, 2)


# Initialize the video capture
cap = cv2.VideoCapture('/home/mahmoud/Desktop/face detection/face_detection/video/v5.mp4')  # Replace 'your_video.mp4' with the video file path

# Create a correlation tracker
tracker = dlib.correlation_tracker()

# Read the first frame
ret, frame = cap.read()

# Select the region of interest (ROI) to track
bbox = cv2.selectROI(frame, False)

# Initialize the tracker with the first frame and ROI
tracker.start_track(frame, dlib.rectangle(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

while True:
    timer = cv2.getTickCount()
    # Read a new frame
    ret, frame = cap.read()

    # Break the loop when we reach the end of the video
    if not ret:
        break

    # Update the tracker
    tracker.update(frame)

    # Get the updated bounding box
    tracked_bbox = tracker.get_position()
    left, top, right, bottom = int(tracked_bbox.left()), int(tracked_bbox.top()), int(tracked_bbox.right()), int(tracked_bbox.bottom())

    drawBox(frame, [left, top, right, bottom], 1)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.imshow('Object Tracking', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
