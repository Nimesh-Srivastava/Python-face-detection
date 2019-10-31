import cv2
import numpy as np

face_detection=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('cris.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_detection.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces :
    img = cv2.rectangle(img,(x,y),(x+w, y+h),(255,0,0),3)
cv2.imwrite('Face_AB.jpg',img)