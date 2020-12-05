import cv2 
from datetime import datetime
import os 

def contains(r1, r2):
    return (r1[0]< r2[0] < r2[0]+r2[2] < r1[0]+r1[2] and r1[1]< r2[1] <r2[1]+r2[3] <r1[1]+r1[3])


os.chdir(r'C:\Users\LTM0110User\Desktop\vs_code\cv00-master\4 - Object Detection')

cap = cv2.VideoCapture(0)
codec= cv2.VideoWriter_fourcc(*'MJPG' ) 
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade=cv2.CascadeClassifier("haarcascade_smile.xml")

out= None
rec= False 
bg_mode= False 
dt_mode= False 
detection= False 

while cap.isOpened():
 
    _, Frame=cap.read()

    if bg_mode:
        Frame= cv2.cvtColor( Frame, cv2.COLOR_BGR2GRAY )
    if dt_mode: 
        strnow= datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        cv2.putText(Frame,strnow, (20, Frame.shape[0]-20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,0), 2 )    
    if rec:
        out.write(Frame)
        cv2.circle(Frame, (Frame.shape[1]-30,Frame.shape[0]-30), 10, (0,0,256),-1  )
    if detection: 
        gray=cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale( gray, 1.05, 9 )
        smiles=smile_cascade.detectMultiScale( gray, 1.5, 30 )
        for f in faces:
            cv2.rectangle(Frame,(f[0],f[1]),(f[0]+f[2],f[1]+f[3]), (0,256,0), 2 )
            for s in smiles:
                if contains(f,s):
                      cv2.rectangle(Frame,(s[0],s[1]),(s[0]+s[2],s[1]+s[3]), (256,0,0), 2 )
                      image= Frame[ f[1]:(f[1]+f[3]),f[0]:(f[0]+f[2]) ] 
                      cv2.imwrite('./res/smiledetected.jpg',image )

    cv2.imshow('video', Frame)
    k= cv2.waitKey(1)
    
    if k== ord("b"):
        bg_mode= not bg_mode 
    elif k== ord("t"):
        dt_mode= not dt_mode
    elif k== ord("c"):
        filename=datetime.now().strftime("%Y%m%d%H%M%S")+ ".jpg" 
        cv2.imwrite(filename, Frame)
        print( 'Immagine catturata' )
    elif k== ord(" "):
        if out== None:
            out= cv2.VideoWriter("output.avi",codec, 20., (640, 480))
        rec= not rec 
        print("registration")
    elif k==ord('r'):
        detection=not detection
    elif k == (ord("q")): 
        break  

if out != None:
    out.release() 

cap.release()
cv2.destroyAllWindows()
