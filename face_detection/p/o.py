import cv2

# Read the video
cap = cv2.VideoCapture('C:/Users/p8036/Desktop/face detection/face_detection/video/v7.mp4')

# Read the first frame
ret, frame = cap.read()

# Set the ROI (Region of Interest)
frame = cv2.resize(frame, (1000, 650), interpolation=cv2.INTER_AREA)
x, y, w, h = cv2.selectROI(frame)

# Initialize the tracker
tracker = cv2.TrackerMOSSE_create()
tracker.init(frame, (x,y,w,h))

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1000, 650), interpolation=cv2.INTER_AREA)
    if not ret:
        break

    # Update the tracker
    ret, track_window = tracker.update(frame)

    # Draw the track window on the frame
    x,y,w,h = int(track_window[0]),int(track_window[1]),int(track_window[2]),int(track_window[3])
    img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()