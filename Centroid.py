import cv2 
import numpy as np
from scipy.spatial import distance as dist
import os 

# This program exploits haarsh-classifiers to perform faces tracking by webcam.   

# Faces are recognized and an index and a centroid are assigned to each of them.
# At each iteration, which repeats every few frames , all the faces are identified by the classifier. 

# If the distance between the centroids of the current faces and the ones recognized in the previous iteration
# is is less than a pre-defined value, the face is recognized as the same, and the same index is given.  


def get_centroid(r):
    return(r[0]+r[2]//2,r[1]+r[3]//2   ) # Centroid formula. r is an array of 4 values each of them representing a side of the rectangle


os.chdir(r'C:\Users\LTM0110User\Desktop\vs_code\cv00-master\5 - Object Tracking'  ) # Setting the fold
cap=  cv2.VideoCapture(0) # Initializing webcam
A, _ = cap.read()

if not A: # if webcam not present, then exit 
    exit(0)

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # pre-builded model
faces_detected= { } # dict containing faces id as key, faces centroids as values
faces_disap={ }
faces_count=0
old_faces=set()
count=0

while cap.isOpened():
    _,Frame= cap.read()
 
    gray= cv2.cvtColor( Frame, cv2.COLOR_BGR2GRAY) # Transforms the image from rgb to grayscale.
    faces= face_cascade.detectMultiScale( gray, 1.04 , 25 ) # Arguments:  Image,  Scalefactor ( Balances accuracy / speed), minNeighbors( Balances Precision )  

    if len(faces) !=0:    # If at least one face is detected
        centroids_found=np.zeros((len(faces),2), dtype="int" )
       
        for i,f in enumerate(faces):
            centroids_found[i]= get_centroid(f) # computing centroids for each faces 
    
        if len( faces_detected)==0: # initialization, no faces still detected
            for i in range(len(centroids_found)):
                    faces_detected[faces_count]= centroids_found[i]  # Assign centroids to the faces 
                    faces_count+=1
            numberframes=np.zeros(len(centroids_found)) # computing how many times the same face has been detected
        else: # Faces have been recognized before this iteration
            count=0    
            faces_ids= list(faces_detected.keys()) 
            faces_centroids= list( faces_detected.values())
            print(np.array(faces_centroids))
            print(centroids_found)
            D=dist.cdist(np.array(faces_centroids),centroids_found) # compute the distance between old faces and new faces
            
            # rows are the IDs of the current faces ordered by priority ( minimum distances)
            # colums are the IDs of the old faces which matches with the current faces. they are ordered by the priority. 
            
            rows=D.min(axis=1).argsort() # Organizing the indices basing on min values along the rows ( managing centroids with the same distance with previus ones  ) 
            cols= D.argmin(axis=1)[rows] # Calculating the index of centroid with the minimum distance, ordering the result by the priority calculated before
          

            used_rows= set()
            used_cols= set()        
            used_id= set()
            # Assigning the values of the centroids to the correct key value(id)
            for (row,col) in zip(rows,cols): 
                if row in used_rows or col in used_cols: # If old or new faces has been already selected, go ahead
                    continue
                face_id= faces_ids[row]
                faces_detected[face_id]= centroids_found[col]
                used_rows.add(row)
                used_cols.add(col)
                used_id.add(face_id)
  

            new_faces= set( range(centroids_found.shape[0]) )- used_cols  

            print(new_faces)
            for new_face in new_faces:
                faces_detected[faces_count] = centroids_found[new_face] # adding new faces to faces_detected
                faces_count+=1
                numberframes=np.append(numberframes,0) 
            
            removed_rows=set()    

            print( faces_ids,numberframes, rows, used_rows)
            No_used=list(set(rows)-used_rows)
            print(No_used) 
            
            # if a face has not been detected for 10 frames, then it is removed
            for N in No_used:
                numberframes[N]+=1
                if numberframes[N]==10:
                    numberframes[N]=0
                    faces_detected.pop(faces_ids[N])
            # Adding to the images the balls ( centroid ) and the rectangles ( faces borders ) 
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

    cv2.imshow('Webcam', Frame) # showing the webcam
 
    k=cv2.waitKey(1)
 
 
    if k==ord("q"): # q is the exit button
        break


cap.release()
cv2.destroyAllWindows()
