import cv2
import math
import sys

def face_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.16, minNeighbors=6, minSize=(30, 30))
    return faces

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def drawBox(frame, bbox, tracker_id, i = 0):
    r = (0, 0, 255)
    if i == 1:
        r = (0, 255, 0)
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3, 1)
    cv2.putText(frame, f"Tracker ID: {tracker_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, r, 2)

cap = cv2.VideoCapture("/home/mahmoud/Desktop/face detection/face_detection/video/v8.mp4")
face_cascade = cv2.CascadeClassifier('/home/mahmoud/Desktop/face detection/face_detection/p/h1.xml')

# Initialize a dictionary to store trackers and their IDs
trackers = {}
tracker_id_counter = 1  # Initialize the tracker ID counter
face_to_tracker_map = {}  # Map detected faces to tracker IDs

ret, frame = cap.read()
i = 0
while True:
    timer = cv2.getTickCount()
    ret, frame = cap.read()
    if not ret:
        break
    
    # Update existing trackers and remove them if necessary
    for tracker_id, (tracker, bbox) in list(trackers.items()):
        success, new_bbox = tracker.update(frame)
        if not success:
            del trackers[tracker_id]
        else:
            trackers[tracker_id] = (tracker, new_bbox)
            drawBox(frame, new_bbox, tracker_id)  
    if i % 4 == 0:
    # Detect faces in the current frame
        faces = face_detection(frame)

        # Initialize new trackers for detected faces if not already tracked
        for face in faces:
            bbox = (face[0], face[1], face[2], face[3])
            face_id = None
            
            # Check if the detected face has been tracked before
            for known_face_id, known_bbox in face_to_tracker_map.items():
                if calculate_distance(bbox[0], bbox[1], known_bbox[0], known_bbox[1]) < 50:
                    face_id = known_face_id
                    break
            
            if face_id is None:
                # Assign a new tracker ID
                face_id = tracker_id_counter
                tracker_id_counter += 1
                
                # Create a new tracker
                tracker = cv2.legacy.TrackerMOSSE_create()
                tracker.init(frame, bbox)
                trackers[face_id] = (tracker, bbox)
            
            face_to_tracker_map[face_id] = bbox
            drawBox(frame, bbox, face_id, 1)
        if cv2.waitKey(1) == ord('q'):
            break
    
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Multi-Object Tracking", cv2.resize(frame, (1000, 650), interpolation=cv2.INTER_AREA))

cap.release()
cv2.destroyAllWindows()
