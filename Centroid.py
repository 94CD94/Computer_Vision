import cv2 
import numpy as np
from scipy.spatial import distance as dist
import os 

def get_centroid(r):
    return(r[0]+r[2]//2,r[1]+r[3]//2   )


os.chdir(r'C:\Users\LTM0110User\Desktop\vs_code\cv00-master\5 - Object Tracking'  )
cap=  cv2.VideoCapture(0)
A, _ = cap.read()

if not A: 
    exit(0)

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faces_detected= { }
faces_disap={ }
faces_count=0
old_faces=set()
count=0
while cap.isOpened():
    _,Frame= cap.read()
 
    gray= cv2.cvtColor( Frame, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale( gray, 1.04 , 25 )

    if len(faces) !=0:    
        centroids_found=np.zeros((len(faces),2), dtype="int" )
       
        for i,f in enumerate(faces):
            centroids_found[i]= get_centroid(f)
    
        if len( faces_detected)==0:
            for i in range(len(centroids_found)):
                    faces_detected[faces_count]= centroids_found[i]
                    faces_count+=1
            numberframes=np.zeros(len(centroids_found))
        else: 
            count=0    
            faces_ids= list(faces_detected.keys())
            faces_centroids= list( faces_detected.values())
            print(np.array(faces_centroids))
            print(centroids_found)
            D=dist.cdist(np.array(faces_centroids),centroids_found) 

            rows=D.min(axis=1).argsort()
            cols= D.argmin(axis=1)[rows]
          

            used_rows= set()
            used_cols= set()        
            used_id= set()

            for (row,col) in zip(rows,cols):
                if row in used_rows or col in used_cols:
                    continue
                face_id= faces_ids[row]
                faces_detected[face_id]= centroids_found[col]
                used_rows.add(row)
                used_cols.add(col)
                used_id.add(face_id)
  

            new_faces= set( range(centroids_found.shape[0]) )- used_cols

            print(new_faces)
            for new_face in new_faces:
                faces_detected[faces_count] = centroids_found[new_face]
                faces_count+=1
                numberframes=np.append(numberframes,0)
            
            removed_rows=set()    

            print( faces_ids,numberframes, rows, used_rows)
            No_used=list(set(rows)-used_rows)
            print(No_used) 

            for N in No_used:
                numberframes[N]+=1
                if numberframes[N]==10:
                    numberframes[N]=0
                    faces_detected.pop(faces_ids[N])

            for row in used_rows :
                face_id= faces_ids[row]
                centroid= faces_detected[face_id]
                cv2.circle(Frame, (centroid[0],centroid[1]) , 8,(0,255,0), -1 )
                cv2.putText(Frame, "ID"+ str(face_id), (centroid[0]-20,centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2 )
    else: 
        count+=1
        if count==15: 
            count=0
            for k in list(faces_detected.keys()):
                faces_detected.pop(k)

    cv2.imshow('Webcam', Frame)
 
    k=cv2.waitKey(1)
 
 
    if k==ord("q"):
        break


cap.release()
cv2.destroyAllWindows()