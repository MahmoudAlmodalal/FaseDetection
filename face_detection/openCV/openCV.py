import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import time


def get_image(dir):
    image_paths = []
    for image in os.listdir(dir):
        image_paths.append(os.path.join(dir, image))
    return image_paths


def show_face_detection(image, faces, elapsed_time, n):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax = plt.gca()
    for x, y, width, height in faces:
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(rect)
    plt.text(10, -10, f"Elapsed time: {elapsed_time:.2f} seconds, {n} faces", fontsize=12, color='blue')
    plt.show()


def face_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(20, 20))
    return faces


if __name__ == "__main__":
    i = 0
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    images = get_image("C:\\Users\\p8036\\Desktop\\face detection\\face_detection\\test_image")
    for image_path in images:
        start_time = time.time()
        image = cv2.imread(image_path)
        faces = face_detection(image)
        i += len(faces)
        end_time = time.time()
        show_face_detection(image, faces, end_time - start_time, len(faces))
        
        # Wait for 'q' key to be pressed, then go to the next image
        key = cv2.waitKey(0)
        if key == ord('q'):
            continue

    print(i)
    cv2.destroyAllWindows()