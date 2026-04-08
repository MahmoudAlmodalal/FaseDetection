import cv2
#30 y

def drawBox(frame, bbox):
    x, y, w, h = [int(i) for i in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw boxes on img



# Create a video capture object
cap = cv2.VideoCapture('/home/mahmoud/Desktop/face detection/face_detection/video/v5.mp4')  # Replace 'video.mp4' with your video file

# Initialize the Mean Shift tracker
tracker = cv2.TrackerMIL.create()

# Read the first frame from the video
ret, frame = cap.read()

# Select a region to track (you can also modify this to use a mouse click)
bbox = cv2.selectROI(frame, False)

# Initialize the tracker with the selected region
tracker.init(frame, bbox)

while True:
    timer = cv2.getTickCount()
    ret, frame = cap.read()
    if not ret:
        break

    # Update the tracker
    ret, bbox = tracker.update(frame)

    drawBox(frame, bbox)
    # Display the frame
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.imshow('Object Tracking', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
