import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mtcnn import MTCNN
import os
import time


def get_image(dir: str) -> list[str]:
    image_paths = []
    for image in os.listdir(dir):
        image_paths.append(dir + "\\" + image)
    return image_paths


def show_face_detection(image, faces_MTCNN, faces_cv, elapsed_time):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax = plt.gca()
    for face in faces_MTCNN:
        x, y, width, height = face['box']
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(rect)
    for x, y, width, height in faces_cv:
        rect = Rectangle((x, y), width, height, fill=False, color='green')
        ax.add_patch(rect)
    plt.text(10, 0, f"Elapsed time: {elapsed_time:.2f} seconds", fontsize=12, color='blue')
    plt.show()


def face_detection_MTCNN(image):
    faces_MTCNN = detector.detect_faces(image)
    return faces_MTCNN


def face_detection_cv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_cv = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
    return faces_cv


if __name__ == "__main__":
    detector = MTCNN(min_face_size=20, steps_threshold=[
                     0.6, 0.7, 0.7], scale_factor=0.7)
    out_time = 0
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    images = get_image(
        "C:\\Users\\p8036\\Desktop\\face detection\\face_detection\\test_image")
    i = 1
    for image_path in images:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping invalid image: {image_path}")
            continue
        start_time = time.time()
        faces_MTCNN = face_detection_MTCNN(image)
        faces_cv = face_detection_cv(image)
        end_time = time.time()
        show_face_detection(image, faces_MTCNN, faces_cv, end_time - start_time)
