import cv2
import math

def face_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.06, minNeighbors=5, minSize=(20, 20))
    return faces

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def drawBox(frame, bbox, tracker_id):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3, 1)
    cv2.putText(frame, f"Tracker ID: {tracker_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cap = cv2.VideoCapture("C:/Users/p8036/Desktop/face detection/face_detection/video/v4.mp4")
face_cascade = cv2.CascadeClassifier('C:/Users/p8036/Desktop/face detection/face_detection/p/h1.xml')

# Create a MultiTracker object to track multiple objects
multi_tracker = cv2.MultiTracker_create()

while True:
    timer = cv2.getTickCount()
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the current frame
    faces = face_detection(frame)

    # Initialize new trackers for detected faces if not already tracked
    for face in faces:
        bbox = (face[0], face[1], face[2], face[3])
        
        # Check if the detected face is already tracked
        already_tracked = False
        for i, tracker in enumerate(multi_tracker.getObjects()):
            tracked_bbox = tracker[i]
            if calculate_distance(bbox[0], bbox[1], tracked_bbox[0], tracked_bbox[1]) < 50:
                already_tracked = True
                break

        if not already_tracked:
            multi_tracker.add(cv2.TrackerMOSSE_create(), frame, bbox)

    # Update the trackers and draw bounding boxes
    success, boxes = multi_tracker.update(frame)
    for i, new_bbox in enumerate(boxes):
        drawBox(frame, new_bbox, i)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Multi-Object Tracking", cv2.resize(frame, (1000, 650), interpolation=cv2.INTER_AREA))

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
