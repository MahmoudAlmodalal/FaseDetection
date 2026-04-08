import dlib
import cv2

# Load the face detector from dlib
detector = dlib.get_frontal_face_detector()

# Load an image
image_path = 'C:\\Users\\p8036\\Desktop\\face detection\\face_detection\\test_image\m.jpeg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray)

# Draw rectangles around the detected faces
for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the result
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()