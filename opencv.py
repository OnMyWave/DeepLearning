import cv2
import numpy as np

fc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #얼굴 찾는 xml
eye = cv2.CascadeClassifier('haarcascade_eye.xml') #눈 찾는 xml

cap = cv2.VideoCapture(0) #카메라 녹화모드
cap.set(3,640) #화질설정
cap.set(4,480)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #회색 변환
    faces = fc.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize=(20,20)) #얼굴 크기
    for (x,y,w,h) in faces: #얼굴에 사각형 표시
        cv2.rectangle(img, (x,y), (x+w, y+h), (100,150,100) ,2)
        roi_gray = gray[y:y+h, x:x+w]

    eyes = eye.detectMultiScale(gray) #눈 찾기
    for(ex, ey, ew, eh) in eyes: #눈 사각형 표시
        cv2.rectangle(img, (ex,ey), (ex+ew, ey+eh), (0,0,255) ,2)
    cv2.imshow('video', img)
    k= cv2.waitKey(30)&0xff #esc 누르면 프로그램 종료
    if k == 27:
        break

cap.release()
cv2.destroyAllwindows()