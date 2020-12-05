#HAAR CASCADE CLASSIFIER 

# Filtri o kernel  ci sono le edge features, le line features e le four rect circles
# Questi quadrati vengono sovrapposti sulla figura. attraverso addizioni si vede se la feature esiste
# Summed area table si aggiunge una riga e una colonna. Per calcolare l'area di una matrice
#
 
# Integral images: nome dato alle summed area table ora le calcoliamo anche se non servono 

import cv2
import os 

os.chdir(r'C:\Users\LTM0110User\Desktop\vs_code\cv00-master\4 - Object Detection'  )
img=cv2.imread('./res/italia.jpg')
integral=cv2.integral(img)

 

# Cascade of classifier o viola jhones algorithm si usa adaboost insieme alle HAAR.
# Ad ogni stage si aumentano le features. Se non si trova l'oggetto ricercato, non si va avanti negli stage 
# Face detection. 

img=cv2.imread('./res/italia.jpg')
img=cv2.resize(img, (img.shape[1]//3, img.shape[0]//3))
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(  gray, 1.1 #Riduco la dim del 5% ogni iterazione, trade off precisione/tempo \
                                            , 20  #Numero di neighbors ( feaures riconosciute) per classificare
                                                # come vero un volto. trade off Precision/recall
                                               ) 
# ritorna una lista di tuple per ogni oggetto trovato.  i valori sono x,y,v,h (si puo disegnare il quadrato
# intorno all'area data)

for (x,y,v,h) in faces: 
    cv2.rectangle( img,(x,y),(x+v,y+h), (0,0,256), 1)

cv2.imshow('detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 

