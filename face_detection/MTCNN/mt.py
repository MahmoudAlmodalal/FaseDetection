import cv2
from deepface import DeepFace

# Load the image using OpenCV
image = cv2.imread("C:\\Users\\p8036\\Desktop\\face detection\\face_detection\\test_image\\m3.jpg")

# Perform face detection using DeepFace
faces = DeepFace.detectFace("C:\\Users\\p8036\\Desktop\\face detection\\face_detection\\test_image\\m3.jpg", detector_backend='mtcnn')

# Create a copy of the original image to draw rectangles on
image_with_rectangles = image.copy()

# Draw a rectangle around each detected face
for face in faces:
    x, y, w, h = map(int, face['box'])
    cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with rectangles around the faces
cv2.imshow('Detected Faces', image_with_rectangles)
cv2.waitKey(0)
cv2.destroyAllWindows()
