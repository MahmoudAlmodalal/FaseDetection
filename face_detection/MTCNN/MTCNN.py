from mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import time


def get_image(dir):
    image_paths = []
    for image in os.listdir(dir):
        image_paths.append(dir + "/" + image)
    return image_paths


def show_face_detection(image, faces, elapsed_time, n):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax = plt.gca()
    for face in faces:
        x, y, width, height = face['box']
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(rect)
    plt.text(10, -10, f"Elapsed time: {elapsed_time:.2f} seconds, {n} faces", fontsize=12, color='blue')
    plt.show()


def face_detection(image, detector):
    faces = detector.detect_faces(image)
    return faces


if __name__ == "__main__":
    detector = MTCNN(min_face_size=20, steps_threshold=[
                     0.513, 0.513, 0.513], scale_factor=0.7)
    images = get_image(
        "/home/mahmoud/Desktop/face detection/face_detection/test_image")
    for image_path in images:
        start_time = time.time()
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping invalid image: {image_path}")
            continue
        faces = face_detection(image, detector)
        n = len(faces)
        end_time = time.time()
        elapsed_time = end_time - start_time
        show_face_detection(image, faces, elapsed_time, n)
