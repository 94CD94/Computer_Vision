import cv2 
import os 

os.chdir(r'C:\Users\LTM0110User\Desktop\vs_code\cv00-master\4 - Object Detection'  )

cap= cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
detection=False

while cap.isOpened():
    _, Frame=cap.read()
    if detection: 
        gray=cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale( gray, 1.05,9 )
        for (x,y,v,h) in faces:
            cv2.rectangle(Frame,(x,y),(x+h,y+v), (0,256,0), 2 )
         
    cv2.imshow('Video',Frame)

    k=cv2.waitKey(1)
    if k==ord("q"):
        break
    elif k==ord('r'):
        detection=not detection
        print("ci sono")


cv2.destroyAllWindows()





