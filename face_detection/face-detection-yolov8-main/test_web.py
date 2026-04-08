from ultralytics import YOLO
import cv2
import math

def drawBox(frame, bbox, face_id, i=0):
    r = (0, 0, 255)
    if i == 1:
        r = (0, 255, 0)
    cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 255, 0), 2)  # draw boxes on img
    cv2.putText(frame, f"Face ID: {face_id}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, r, 5)

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Initialize YOLOv8 model
model = YOLO("/home/mahmoud/Desktop/face detection/face_detection/face-detection-yolov8-main/yolov8n-face.pt")
# Open a video capture
cap = cv2.VideoCapture("/home/mahmoud/Desktop/face detection/face_detection/video/v1.mp4")

# Initialize face ID counter
face_id_counter = 1
face_to_tracker_map = {}

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
        boxes = result.boxes.cpu().numpy()  # get boxes on CPU in numpy
        for box in boxes:  # iterate boxes
            r = box.xyxy[0].astype(int)  # get corner points as int

            # Check if the face is already tracked
            matched_face_id = None
            for face_id, bbox in face_to_tracker_map.items():
                x1, y1, x2, y2 = bbox
                center_x = r[0] + (r[2] - r[0]) / 2
                center_y = r[1] + (r[3] - r[1]) / 2

                # Calculate distance between centers
                distance = calculate_distance(center_x, center_y, (x1 + x2) / 2, (y1 + y2) / 2)
                if distance < frame.shape[1] * frame.shape[0] * 0.000017:  # You can adjust this threshold as needed
                    print(f"({x1}, {y1}) ({x2}, {y2}) - ({r[0]}, {r[1]}) ({r[2]}, {r[3]})")
                    print(f"h{x2 - x1}, w{y2 - y1}, h{r[2] - r[0]}, w{r[3] - r[1]}")
                    matched_face_id = face_id
                    break

            if matched_face_id is None:
                # Assign a new face ID
                matched_face_id = face_id_counter
                face_id_counter += 1

            face_to_tracker_map[matched_face_id] = (r[0], r[1], r[2], r[3])
            drawBox(frame, r, matched_face_id)
    
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    # Display the frame with detected objects using OpenCV
    cv2.imshow("YOLOv8 Object Detection", cv2.resize(frame, (int(frame.shape[1] * 0.3), int(frame.shape[0] * 0.3)), interpolation=cv2.INTER_AREA))

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()
