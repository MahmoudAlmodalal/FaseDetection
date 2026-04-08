import cv2
import face_recognition
import glob
folder_dir = 'images'


image1=cv2.imread("images/cristiano ronaldo 1.jpg")
rgbimage1=cv2.cvtColor(image1,  cv2.COLOR_BGR2RGB)
img_encoding1=face_recognition.face_encodings(rgbimage1)[0]

for images in glob.iglob(f'{folder_dir}/*'):

    
    if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg") ):
        print(images)
        image2 = cv2.imread(images)
        rgbimage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        img_encoding2 = face_recognition.face_encodings(rgbimage2)[0]
        result = face_recognition.compare_faces([img_encoding1], img_encoding2)
        print("Result", result)
        cv2.imshow("image", image1)
        cv2.imshow("image 2", image2)
        cv2.waitKey(0)